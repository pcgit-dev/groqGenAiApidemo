#Tools creation
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool


#Used inbuilt tool of wikipedia
api_wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
tool_wiki=WikipediaQueryRun(api_wrapper=api_wiki_wrapper)

api_arxivwrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
tool_arxiv=ArxivQueryRun(api_wrapper=api_arxivwrapper)

tools=[tool_wiki,tool_arxiv]

loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectorDb=FAISS.from_documents(documents,OpenAIEmbeddings())
retriever=vectorDb.as_retriever()

retriver_tool=create_retriever_tool(retriever,"langscmith search","Search any information about langsmith")

print(retriver_tool.name)
