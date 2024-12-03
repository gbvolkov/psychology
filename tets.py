import os
import re

import config
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.gigachat import GigaChat
#from yandex_chain import YandexLLM
from langchain_community.llms import YandexGPT

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, Tuple

import logging
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
        , logging.FileHandler('./output/kb_retriever.log')
    ],
)

os.environ["LANGCHAIN_TRACING_V2"] = "true"


class ChatAssistant(ABC):
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize the chat assistant with an optional system prompt.

        :param system_prompt: A string representing the system prompt to guide the assistant.
        """
        logging.info("Initializing chat model")
        self.llm = self.initialize()
        self.conversation_history = []
        if system_prompt:
            self.set_system_prompt(system_prompt)
        else:
            logging.warning("No system prompt provided. The assistant may not behave as expected.")

        logging.info("Initialized")

    @abstractmethod
    def initialize(self):
        """
        Initialize the chat model here.
        Should return an instance of a LangChain chat model.
        """
        pass

    def set_system_prompt(self, prompt: str):
        """
        Set or update the system prompt for the assistant.

        :param prompt: The system prompt string.
        """
        self.system_prompt = prompt
        if self.conversation_history and self.conversation_history[0]['role'] == 'system':
            self.conversation_history[0]['content'] = prompt
            logging.info("System prompt updated.")
        else:
            # Insert the system prompt at the beginning of the conversation history
            self.conversation_history.insert(0, {"role": "system", "content": prompt})
            logging.info("System prompt set.")

    def run_chain(self, template, input, llm):
        input_variables = list(input.keys())

        prompt = PromptTemplate(
            input_variables=input_variables,
            template=template,
        )
        chain = prompt | llm
        return chain.invoke(input).content

    def ask_question(self, query: str) -> str:
        """
        Send a user query to the assistant and get the response.

        :param query: The user's question or message.
        :return: The assistant's response as a string.
        """
        if self.llm is None:
            logging.error("LLM not initialized")
            raise ValueError("Model not initialized.")

        try:
            history = "".join([f"{item['role']}: {item['content']}\n\n" for item in self.conversation_history[1:]])
            # Append user query to the conversation history

            while True:
                try:
                    # Get the response from the model
                    content = self.run_chain(
                        template=self.system_prompt,
                        input={"USER_INPUT": query, "CONVERSATION_HISTORY": history},
                        llm=self.llm    
                    )
                    break
                except Exception as e:
                    logging.error(f"Error in ask_question: {str(e)}")

            

            self.conversation_history.append({"role": "user", "content": query})
            logging.debug(f"Added user message to history: {query}")
            #while True:
            #    try:
            #        response = self.llm.invoke(self.conversation_history)
            #        logging.debug(f"Received response from model: {response}")
            #        break
            #    except Exception as e:
            #        logging.error(f"Error in ask_question: {str(e)}")

            # Append assistant's response to the conversation history
            match = re.search(r"<response>(.*?)</response>", content, re.DOTALL)
            
            if match:
                response = match.group(1).strip()
            else:
                response = content.strip()
            self.conversation_history.append({"role": "assistant", "content": response})
            logging.debug(f"Added assistant message to history: {content}")

            return content
        except Exception as e:
            logging.error(f"Unexpected error in ask_question: {str(e)}")
            raise
    
    def add_assistant_context(self, context: str):
        self.conversation_history.append({"role": "system", "content": context})

    def add_conversation_history(self, conversation_history = []):
        self.conversation_history.extend(conversation_history)


class ChatAssistantGPT(ChatAssistant):
    def __init__(self, system_prompt: Optional[str] = None):
        super().__init__(system_prompt)

    def initialize(self):
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4)    

MISTRAL_FT_MODEL = "ft:mistral-small-latest:08555483:20241130:eaf9fe5a"
class ChatAssistantMistralAI(ChatAssistant):
    def __init__(self, system_prompt: Optional[str] = None):
        super().__init__(system_prompt)

    def initialize(self):
        return ChatMistralAI(
            #model="open-mistral-nemo",
            model=MISTRAL_FT_MODEL,
            api_key=config.MISTRAL_API_KEY,
            temperature=0.4,
            top_p=0.7)
    

from new import supervisor_prompt, assistant_prompt, user_prompt

# Initialize the assistant with the system prompt
assistant = ChatAssistantMistralAI(system_prompt=assistant_prompt)
supervisor = ChatAssistantGPT(system_prompt=supervisor_prompt)

user = ChatAssistantGPT(system_prompt=user_prompt)
idx: int = 1


#logging = logging.getlogging(__name__)

assistant_input = "Здравствуйте. Чем я могу Вам помочь?"
while True:
    recommedation: str = "<supervisor_report><summary></summary><emotional_state></emotional_state><recommendation></recommendation></supervisor_report>"
    #user_input = input("You: ")
    user_input = user.ask_question(assistant_input)
    logging.info(f"==========================\nUser: {user_input}\n\n")
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chat. Goodbye!")
        break
    if len(assistant.conversation_history) >= 110000:
        history = assistant.conversation_history[1:]
        supervisor.add_conversation_history(history[-12:])
        #supervisor.conversation_history.append({"role": "user", "content": user_input})
        #response = supervisor.ask_question('Please, assess the conversation')
        supervisor_input = supervisor.ask_question(user_input)
        recommedation = supervisor_input.content
        logging.info(f"Supervisor: {recommedation}\n\n")
        #assistant.set_system_prompt(f'{assistant_prompt}\n\nРекомендации супервизора: \n {response.content}')
        del assistant.conversation_history[2:7]
    
    #if len(assistant.conversation_history) > 2:
    #    last_item = assistant.conversation_history[-2]
    #    if last_item['role'] == 'user':
    #        user_input = last_item['content'].split('\n\n## Рекомендации супервайзора:', 1)[0]
    #        assistant.conversation_history[-2]['content'] = user_input
    user_message = {"role": "user", "content": user_input} 
    assistant_input = assistant.ask_question(user_input) # \n\n## Отчёт супервизора: {recommedation}")
    logging.info(f"Assistant: {assistant_input}\n\n\n\n")
