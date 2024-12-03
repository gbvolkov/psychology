import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting bot...")

import telebot
from telebot import types
import uuid
import json
import os
#from vrecog.vrecog import recognise_text
#from storage_managers.google_sheets_man import store_shipment
from typing import List, Any, Optional, Dict, Tuple

from assistants.assistants import JSONAssistantGPT
import config

bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)

# Хранилище данных пользователей
user_data = {}

# Состояния пользователей
class UserState:
    IDLE = 'idle'
    ADDING_SHIPMENT = 'adding_shipment'
    CONFIRMING_SHIPMENT = 'confirming_shipment'
    CORRECTING_FIELD = 'correcting_field'
    AWAITING_NEXT_STEP = 'awaiting_next_step'

# Поля для корректировки
shipment_fields = [
    "shipment_date",
    "shipment_time",
    "customer_name",
    "customer_address",
    "good",
    "good_volume",
    "good_price",
    "shipment_count",
    "shipment_cost",
    "supplier"
]

procurement_fields = [
    "supplier",
    "good",
    "good_volume",
    "good_price",
    "supply_cost"
]

def parse_shipment(text):
    from typing import Optional
    from pydantic import BaseModel, Field

    class Procurement(BaseModel):
        supplier: Optional[str] = Field(default=None)
        good: Optional[str] = Field(default=None)
        good_volume: Optional[str] = Field(default=None)
        good_price: Optional[str] = Field(default=None)
        supply_cost: Optional[str] = Field(default=None)

    class Shipment(BaseModel):
        shipment_date: Optional[str] = Field(default=None)
        shipment_time: Optional[str] = Field(default=None)
        customer_name: Optional[str] = Field(default=None)
        customer_address: Optional[str] = Field(default=None)
        good: Optional[str] = Field(default=None)
        good_volume: Optional[str] = Field(default=None)
        good_price: Optional[str] = Field(default=None)
        shipment_count: Optional[str] = Field(default=None)
        shipment_cost: Optional[str] = Field(default=None)
        supplier: Optional[str] = Field(default=None)
        procurements: Optional[List[Procurement]] = Field(default=None)

    class Shipments(BaseModel):
        shipments: List[Shipment] = Field(description="List of shipments")

    assistant = JSONAssistantGPT(schema=Shipments)
    result = assistant.ask_question(text)
    return json.loads(result).get('shipments')

def initialize_user(user_id):
    if user_id not in user_data:
        user_data[user_id] = {
            'state': UserState.IDLE,
            'shipments': [],
            'shipment_queue': [],
            'current_shipment_index': 0,
            'current_field': None,
            'procurement': {},
            'shipment_id': None
        }

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Добро пожаловать! Используйте команду /add_shipment для добавления новой отгрузки.")

@bot.message_handler(commands=['add_shipment'])
def add_shipment(message):
    user_id = message.from_user.id
    initialize_user(user_id)
    user_data[user_id]['state'] = UserState.ADDING_SHIPMENT
    bot.send_message(user_id, "Пожалуйста, отправьте информацию о отгрузке в виде текста или голосового сообщения.")

@bot.message_handler(content_types=['text', 'voice'])
def handle_message(message):
    user_id = message.from_user.id
    initialize_user(user_id)
    state = user_data[user_id]['state']

    if state == UserState.ADDING_SHIPMENT:
        handle_adding_shipment(message, user_id)
    elif state == UserState.CONFIRMING_SHIPMENT:
        handle_confirming_shipment(message, user_id)
    elif state == UserState.CORRECTING_FIELD:
        handle_correcting_field(message, user_id)
    elif state == UserState.AWAITING_NEXT_STEP:
        handle_next_step(message, user_id)
    else:
        bot.send_message(user_id, "Для добавления новой отгрузки используйте команду /add_shipment.")

def handle_adding_shipment(message, user_id):
    if message.content_type == 'voice':
        bot.send_message(user_id, "Распознаю голосовое сообщение...")
        try:
            file_info = bot.get_file(message.voice.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            ogg_file_path = f"voice_{user_id}_{uuid.uuid4()}.ogg"
            with open(ogg_file_path, 'wb') as f:
                f.write(downloaded_file)
            
            # Передаём путь к OGG-файлу в recognise_text
            text = recognise_text(ogg_file_path)
            os.remove(ogg_file_path)
            
            if not text:
                bot.send_message(user_id, "Не удалось распознать голосовое сообщение. Пожалуйста, отправьте текст вручную или попробуйте снова.")
                user_data[user_id]['state'] = UserState.IDLE
                return
            logger.info(f"Распознанный текст:\n{text}")
        except Exception as e:
            logger.error(f"Error processing voice message: {str(e)}")
            bot.send_message(user_id, "Не удалось распознать голосовое сообщение. Пожалуйста, отправьте текст вручную или попробуйте снова.")
            user_data[user_id]['state'] = UserState.IDLE
            return
    else:
        text = message.text

    try:
        shipments = parse_shipment(text)
        if not shipments:
            bot.send_message(user_id, "Не удалось разобрать информацию о отгрузке. Пожалуйста, проверьте формат и попробуйте снова.")
            user_data[user_id]['state'] = UserState.IDLE
            return
        user_data[user_id]['shipments'] = shipments  # Original list
        #user_data[user_id]['shipment_queue'] = shipments.copy()  # Clone for queue processing
        user_data[user_id]['current_shipment_index'] = 0
        user_data[user_id]['state'] = UserState.CONFIRMING_SHIPMENT
        send_shipment_confirmation(user_id)
    except Exception as e:
        logger.error(f"Error processing shipment: {str(e)}")
        bot.send_message(user_id, "Не удалось разобрать информацию о отгрузке. Пожалуйста, проверьте формат и попробуйте снова.")
        user_data[user_id]['state'] = UserState.IDLE

def send_shipment_confirmation(user_id):
    queue = user_data[user_id]['shipments']
    index = user_data[user_id]['current_shipment_index']
    
    if index >= len(queue):
        bot.send_message(user_id, "Все отгрузки обработаны.")
        user_data[user_id]['state'] = UserState.IDLE
        return
    
    current_shipment = queue[index]
    user_data[user_id]['shipment'] = current_shipment  # Set current shipment for processing

    confirmation_text = "Пожалуйста, подтвердите информацию об отгрузке:\n\n"
    confirmation_text += f"Дата отгрузки: {current_shipment.get('shipment_date', '')}\n"
    confirmation_text += f"Время отгрузки: {current_shipment.get('shipment_time', '')}\n"
    confirmation_text += f"Наименование грузополучателя: {current_shipment.get('customer_name', '')}\n"
    confirmation_text += f"Адрес грузополучателя: {current_shipment.get('customer_address', '')}\n"
    confirmation_text += f"Наименование товара: {current_shipment.get('good', '')}\n"
    confirmation_text += f"Объём/количество товара: {current_shipment.get('good_volume', '')}\n"
    confirmation_text += f"Цена товара: {current_shipment.get('good_price', '')}\n"
    confirmation_text += f"Количество отгрузок: {current_shipment.get('shipment_count', '')}\n"
    confirmation_text += f"Стоимость отгрузки: {current_shipment.get('shipment_cost', '')}\n"
    confirmation_text += f"Наименование поставщика: {current_shipment.get('supplier', '')}\n"
    
    procurements = current_shipment.get('procurements', [])
    procurements = [] if procurements is None else procurements
    for procurement in procurements:
        confirmation_text += '--------------------------\n'
        confirmation_text += f"Поставка:\n"
        confirmation_text += f"\tНаименование поставщика: {procurement.get('supplier', '')}\n"
        confirmation_text += f"\tНаименование товара: {procurement.get('good', '')}\n"
        confirmation_text += f"\tОбъём/количество товара: {procurement.get('good_volume', '')}\n"
        confirmation_text += f"\tЦена товара: {procurement.get('good_price', '')}\n"
        confirmation_text += f"\tСтоимость поставки: {procurement.get('supply_cost', '')}\n"
    confirmation_text += '===============================\n'

    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    markup.add('Сохранить', 'Исправить')
    bot.send_message(user_id, confirmation_text, reply_markup=markup)

def handle_confirming_shipment(message, user_id):
    response = message.text.strip().lower()
    if response == 'да':
        shipment = user_data[user_id]['shipment']
        shipment_id = str(uuid.uuid4())
        shipment['shipment_id'] = shipment_id
        try:
            store_shipment(json.dumps(shipment))
            bot.send_message(user_id, f"Отгрузка сохранена с ID: {shipment_id}")
            # Move to the next shipment
            user_data[user_id]['current_shipment_index'] += 1
            if user_data[user_id]['current_shipment_index'] < len(user_data[user_id]['shipments']):
                send_shipment_confirmation(user_id)
            else:
                offer_next_steps(user_id)
                user_data[user_id]['state'] = UserState.AWAITING_NEXT_STEP
        except Exception as e:
            logger.error(f"Error storing shipment: {str(e)}")
            bot.send_message(user_id, "Не удалось сохранить отгрузку. Пожалуйста, попробуйте ещё раз.")
            user_data[user_id]['state'] = UserState.IDLE
    elif response == 'нет':
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        for field in shipment_fields:
            markup.add(translate_field(field))
        bot.send_message(user_id, "Какое поле вы хотите исправить?", reply_markup=markup)
        user_data[user_id]['state'] = UserState.CORRECTING_FIELD
    else:
        bot.send_message(user_id, "Пожалуйста, выберите 'Да' или 'Нет'.")

def handle_correcting_field(message, user_id):
    field = translate_field_to_key(message.text)
    if field and field in shipment_fields:
        user_data[user_id]['current_field'] = field
        bot.send_message(user_id, f"Введите новое значение для '{translate_field(field)}':")
        # Register the next step handler
        bot.register_next_step_handler(message, process_field_correction, user_id)
    else:
        bot.send_message(user_id, "Некорректное поле. Пожалуйста, выберите из предложенных вариантов.")

def process_field_correction(message, user_id):
    field = user_data[user_id]['current_field']
    new_value = message.text
    user_data[user_id]['shipment'][field] = new_value
    bot.send_message(user_id, f"Поле '{translate_field(field)}' обновлено на '{new_value}'.")
    user_data[user_id]['state'] = UserState.CONFIRMING_SHIPMENT
    send_shipment_confirmation(user_id)

def offer_next_steps(user_id):
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    markup.add("Добавить новую отгрузку")
    bot.send_message(user_id, "Что вы хотите сделать дальше?", reply_markup=markup)

def handle_next_step(message, user_id):
    response = message.text.strip()
    if response == "Добавить новую отгрузку":
        add_shipment(message)
    else:
        bot.send_message(user_id, "Пожалуйста, выберите один из предложенных вариантов.")

def handle_adding_procurement(message, user_id):
    field = user_data[user_id]['current_field']
    user_data[user_id]['procurement'][field] = message.text
    if next_field := get_next_field(procurement_fields, field):
        user_data[user_id]['current_field'] = next_field
        bot.send_message(user_id, f"Пожалуйста, введите '{translate_field(next_field)}':")
    else:
        # Все поля закупки введены
        procurement = user_data[user_id]['procurement']
        add_procurement_to_shipment(user_data[user_id]['shipment_id'], procurement, user_id)
        bot.send_message(user_id, "Закупка успешно добавлена.")
        user_data[user_id]['procurement'] = {}
        offer_next_steps(user_id)
        user_data[user_id]['state'] = UserState.AWAITING_NEXT_STEP

def add_procurement_to_shipment(shipment_id, procurement, user_id):
    try:
        shipment = user_data[user_id]['shipment']
        if 'procurements' not in shipment:
            shipment['procurements'] = []
        shipment['procurements'].append(procurement)
        store_shipment(shipment)
    except Exception as e:
        logger.error(f"Error adding procurement to shipment: {str(e)}")
        bot.send_message(user_id, "К сожалению, не могу добавить закупку. Пожалуйста, попробуйте ещё раз.")

def translate_field(field_key):
    translations = {
        "shipment_date": "Дата отгрузки",
        "shipment_time": "Время отгрузки",
        "customer_name": "Наименование грузополучателя",
        "customer_address": "Адрес грузополучателя",
        "good": "Наименование товара",
        "good_volume": "Объём/количество товара",
        "good_price": "Цена товара",
        "shipment_count": "Количество отгрузок",
        "shipment_cost": "Стоимость отгрузки",
        "supplier": "Наименование поставщика",
        "procurement": "Закупка",
        "id": "ID"
    }
    return translations.get(field_key, field_key)

def translate_field_to_key(field_name):
    translations = {
        "Дата отгрузки": "shipment_date",
        "Время отгрузки": "shipment_time",
        "Наименование грузополучателя": "customer_name",
        "Адрес грузополучателя": "customer_address",
        "Наименование товара": "good",
        "Объём/количество товара": "good_volume",
        "Цена товара": "good_price",
        "Количество отгрузок": "shipment_count",
        "Стоимость отгрузки": "shipment_cost",
        "Наименование поставщика": "supplier",
        "Закупка": "procurement",
        "ID": "id"
    }
    return translations.get(field_name)

def get_next_field(fields, current_field):
    try:
        index = fields.index(current_field)
        return fields[index + 1]
    except (ValueError, IndexError):
        return None

if __name__ == '__main__':
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    logger.info("Bot started...")
    bot.infinity_polling()
