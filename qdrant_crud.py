from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.schema import  Document
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import os
import openai

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def get_docstore(collection_name):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        tiktoken_model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        dimensions=256
    )

    client = QdrantClient(host="localhost", port=6333, timeout=3600)

    doc_store = Qdrant(
        client=client, collection_name=collection_name, embeddings=embeddings
    )
    return doc_store

def collection_exists(doc_store: Qdrant):
    list_collections = [c.name for c in doc_store.client.get_collections().collections]
    return doc_store.collection_name in list_collections

def recreate_collection(
    collection_name: str,
    docs_list: list[Document],
    payload_indexes: list[str]
):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        tiktoken_model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        dimensions=256
    )
    doc_store = Qdrant.from_documents(
        documents=docs_list,
        embedding=embeddings,
        host="localhost",
        port=6333,
        force_recreate=True,
        collection_name=collection_name,
    )

    for i in payload_indexes:
        doc_store.client.create_payload_index(
            collection_name, field_name=i, field_schema="keyword"
        )
    return payload_indexes

def filename_exists(file_name, collection_name="knowledge-base"):
    doc_store = get_docstore(collection_name=collection_name)
    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(
                    value=file_name,
                ),
            )
        ]
    )
    try:
        indexes, _ = doc_store.client.scroll(
            collection_name=collection_name, scroll_filter=query_filter, limit=1
        )
        return len(indexes) != 0
    except:
        return False
    
def upload_documents(
    documents: list[Document], source=None, collection_name="knowledge-base"
):
    doc_store = get_docstore(collection_name=collection_name)
    if not collection_exists(doc_store):
        indexes = recreate_collection(
            collection_name=collection_name,
            docs_list=documents,
            payload_indexes=["metadata.source", "metadata.page"]
        )
        return len(indexes) != 0
    else:
        if filename_exists(source, collection_name="knowledge-base"):
            return True
        else:
            indexes = doc_store.add_documents(
                documents=documents,
            )
            return len(indexes) != 0

def doc_retriever(source, collection_name="knowledge-base",limit=5):
    doc_store = get_docstore(collection_name=collection_name)
    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(
                    value=source,
                ),
            )
        ]
    )
    return doc_store.as_retriever(
        search_type="similarity",
        search_kwargs={"filter": query_filter,"k": limit},
    )