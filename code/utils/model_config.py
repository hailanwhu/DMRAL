from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import json
import re


base_url = 'https://api.openai.com/v1'
api_key = 'configure your own keys'
model = 'gpt-4.1-mini'

llm = ChatOpenAI(
    model=model,
    openai_api_base=base_url,
    openai_api_key=api_key,
    temperature=0.3
)
