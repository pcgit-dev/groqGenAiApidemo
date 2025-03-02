from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq 

load_dotenv()

#Load groq api key
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ['GROQ_API_KEY'] = groq_api_key

llm = ChatGroq(groq_api_key=groq_api_key,model="llama3-8b-8192")
promt_template = """
    Define the given term in 25 words
    term : {term}
    no need for preamble. Just provide the definition.
    """
define_promt_temp=PromptTemplate(
    template=promt_template,
    input_variables=['term']
)
llm = ChatGroq(groq_api_key=groq_api_key,model="llama3-8b-8192")

define_term_chain = define_promt_temp | llm
response = define_term_chain.invoke({"term":"Reliablity"})


definition_scoring_template = """
You are given the definition of a term. Your job is to score the definition on a scale of 0 to 5.

definition: {definition}

no need to preamble. Just provide the score.
"""

definition_scoring_prompt_template = PromptTemplate(
                        template=definition_scoring_template,
                        input_variables = ["definition"]
                    )

definition_scoring_chain = definition_scoring_prompt_template | llm

sequential_chain = {"definition": define_term_chain} | definition_scoring_chain

response = sequential_chain.invoke({"term": "momentum"})

print(response)