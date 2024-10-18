from langchain_community.llms import OpenAI
from pydantic import BaseModel
from utils import MEM_DIR, retrieve_chat_history
import langchain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from qdrant_crud import doc_retriever
import os
import openai
from dotenv import load_dotenv
load_dotenv()

# Configure langchain cache
langchain.llm_cache = SQLiteCache(database_path=f"{MEM_DIR}.langchain.db")
# API Key for OpenAI
OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")

def search_docs(query: str, source, chat_id=None):
    question_rephrase_template = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
        Given the following conversation, and a follow-up question, rephrase the follow-up question to be a standalone question.
        Chat History:
        {chat_history}

        Follow Up Input:
        {question}

        Standalone question:
        """
    )
    prompt_template = (
        "Use the following pieces of context to answer the question at the end. \n"
        "context: {context}\n\n"
        "Question: {question}\n"
        "Helpful Answer:"
    )

    retriever = doc_retriever(source=source, limit=5)
    memory, chat_id = retrieve_chat_history(chat_id=chat_id)

    llm = ChatOpenAI(
        cache=True,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo",
    )
    condense_question_llm = ChatOpenAI(
        cache=True,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo",
    )
    question_generator = LLMChain(
        llm=condense_question_llm,
        prompt=question_rephrase_template,
    )

    doc_chain = load_qa_chain(
    llm=llm,
    chain_type="stuff",
    prompt=PromptTemplate.from_template(prompt_template)
    )

    qa = ConversationalRetrievalChain(
    retriever=retriever,
    memory= memory,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True   
    )

    result = qa({"question": query, "chat_history": memory})
    page_sources = [
        f'pg: {doc.metadata.get("page") + 1}'
        for doc in result.get("source_documents", [])
    ]
    page_sources_str = "\n\nSources: " + ", ".join(page_sources)

    response = f'{result.get("answer")}{page_sources_str}'
    return response, chat_id

class QueryIn_unary(BaseModel):
    query: str
    chat_id: str | None = None
    file_name: str
