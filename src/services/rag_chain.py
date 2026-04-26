import os
from dotenv import load_dotenv
from fastapi import Depends
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from src.infrastructure.constants import Topic
from src.infrastructure.models import MathFacultyRequest
from src.infrastructure.models import MathFacultyRequest
from src.services.retriever import get_vector_store, get_vector_store_buk, get_vector_store_qa
from src.infrastructure.prompt_templates import MATH_FACULTY_GENERAL, QA_HELPER, ROMANIAN_CULTURE_HELPER
from langchain.retrievers.multi_query import MultiQueryRetriever
from src.services.retrieval_enhanced import retrieve_with_metadata_boost, format_structured_context

load_dotenv()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ['LANGCHAIN_API_KEY'] = langchain_api_key
if OPEN_API_KEY:
    os.environ['OPENAI_API_KEY'] = OPEN_API_KEY


def query_math_faculty(question: str, chat_history: list[str], vector_store = Depends(get_vector_store)):
    rag_chain = create_rag_chain(MATH_FACULTY_GENERAL, vector_store)
    return rag_chain.invoke(question)

def query_qa(question: str, chat_history: list[str], vector_store = Depends(get_vector_store_qa)):
    rag_chain = create_rag_chain(QA_HELPER, vector_store)
    return rag_chain.invoke(question)

def query_romanian_culture(question: str, chat_history: list[str], vector_store = Depends(get_vector_store_buk)):
    rag_chain = create_rag_chain(ROMANIAN_CULTURE_HELPER, vector_store)
    return rag_chain.invoke(question)

def create_rag_chain(template: ChatPromptTemplate, vector_store = Depends(get_vector_store)):
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    # Multi-Query Retriever: It re-writes the user query into 3 versions
    # to catch relevant chunks even if the wording is different.
    base_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,              # ← GET MORE CANDIDATES
            "score_threshold": 0.5  # ← LOWERED from 0.7 for better recall
        }
    )
    
    # Add error handling for MultiQueryRetriever
    try:
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, 
            llm=llm
        )
    except Exception as e:
        print(f"⚠️  MultiQueryRetriever failed: {e}, using base retriever")
        retriever = base_retriever

    def format_docs(docs):
        """Format documents with structure preservation."""
        if not docs:
            return "No relevant context found."
        # Use enhanced formatting that groups related info
        return format_structured_context(docs)
    
    rag_chain = (
        {
            "context": retriever | format_docs,  # ← Format docs to string
            "question": RunnablePassthrough()
        }
        | template
        | llm
        | StrOutputParser()
    )

    return rag_chain