import os
import re
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, Tuple

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.llms import YandexGPT

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

import config  # Ensure this module securely stores your API keys
from new import supervisor_prompt, assistant_prompt, user_prompt  # Ensure these are predefined prompts

import tiktoken  # Ensure tiktoken is installed

# ===============================
# 1. Logging Configuration
# ===============================
import sys

# Constants
MISTRAL_FT_MODEL = "ft:mistral-small-latest:08555483:20241130:eaf9fe5a"
LOG_FILE_PATH = './output/conversation.log'
sys.stdout.reconfigure(encoding='utf-8')
# Set up logging once at the beginning
#logging.basicConfig(
#    level=logging.INFO,
#    format='%(asctime)s - %(levelname)s - %(message)s',
#    handlers=[
#        logging.StreamHandler(sys.stdout),  
#        logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
#    ],
#)


logger = logging.getLogger('conversation')
logger.setLevel(logging.INFO)  # Set the desired logging level

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create and configure StreamHandler for console output
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)  # You can set different levels if needed
stream_handler.setFormatter(formatter)

# Create and configure FileHandler for file output
file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
file_handler.setLevel(logging.INFO)  # Ensure this matches the logger's level
file_handler.setFormatter(formatter)

# Add handlers to the custom logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# Prevent log messages from being propagated to the root logger
logger.propagate = False


# ===============================
# 2. Define the State Schema
# ===============================

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    language: Optional[str]
    system_prompt: Optional[str]  # To store the system prompt if updated

# ===============================
# 3. Define the Prompt Template
# ===============================

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "{system_prompt}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

psychologist_model = ChatMistralAI(
            #model="open-mistral-nemo",
            model=MISTRAL_FT_MODEL,
            api_key=config.MISTRAL_API_KEY,
            temperature=0.6,
            top_p=0.7)

user_model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.9)  

# ===============================
# 6. Define the Call Model Function
# ===============================


USER_SYSINT = (
    "system",
    user_prompt,
)
# Define the function that calls the model
def user_node(user_state: MessagesState):
    new_output = user_model.invoke([USER_SYSINT] + user_state["messages"])
    logger.info(f'User: {new_output.content}\n\n')
    return user_state | {"messages": [("user", new_output.content)]}

WELCOME_MSG = "Здравствуйте! Чем я могу Вам помочь?"
PSYCHOLOGIST_SYSINT = (
    "system",
    assistant_prompt,
)


def get_response(text): 
    response_text = text
    try:
        response_text = text.split('<response>')[1].split('</response>')[0]
        pattern = re.compile(r'<response>(.*?)</response>', re.DOTALL | re.IGNORECASE)
        # Search for the pattern in the XML data
        match = pattern.search(text)
        if match:
            response_text = match.group(1).strip()
    finally:
        return response_text


def psychologist_node(psyco_state: MessagesState):
    if psyco_state["messages"]:
        new_output = psychologist_model.invoke([PSYCHOLOGIST_SYSINT] + psyco_state["messages"])
    else:
        new_output = AIMessage(content=WELCOME_MSG)
    logger.info(f'Assistan: {get_response(new_output.content)}\n==============================================\n\n')
    return psyco_state | {"messages": [new_output]}

# ===============================
# 7. Initialize LangGraph Workflow
# ===============================

# Initialize memory saver (using in-memory for simplicity; replace with persistent storage as needed)
memory = MemorySaver()

# Define the workflow using StateGraph
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "psychologist")
workflow.add_edge("psychologist","user")
workflow.add_edge("user", "psychologist")
workflow.add_node("user", user_node)
workflow.add_node("psychologist", psychologist_node)

# Compile the application with the workflow and memory
app = workflow.compile(checkpointer=memory)

config = {"recursion_limit": 100, "configurable": {"thread_id": "abc345"}}

#query = ""

input_messages = []
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
