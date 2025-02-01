import os
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAI
from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes



load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)

messages = [
            SystemMessage(
                content="Translate English to french"
            ),
            HumanMessage(
                content="What is your name?"
            )
        ]
   
result=model.invoke(messages)
parser = StrOutputParser()


chain = model|parser
results=chain.invoke(messages)

print(results)

