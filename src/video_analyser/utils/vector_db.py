import chromadb
from pathlib import Path
from pydantic import BaseModel  
from datetime import datetime
import json
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any, Optional


EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

class VideoDescriptionMetadata(BaseModel):
    id: str
    file_name: str
    folder_name: str
    video_description: str
    location: str
    created: str
    duration: int

def create_embedding(text:str, embedding_model=EMBEDDING_MODEL):
    return embedding_model.encode(text).tolist()

def init_chromadb():
    # Initialize ChromaDB client (Persistent storage can be enabled by specifying a path)
    chroma_db_path = Path( "./chromadb")
    chroma_db_path.mkdir(exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))  # Use chromadb.EphemeralClient() for in-memory
    os.chmod(str(chroma_db_path), 0o777)
    collection = chroma_client.get_or_create_collection(name="video_description")
    return collection

def write_video_description_to_vector_db(collection, embedding: list, metadata: VideoDescriptionMetadata):
    collection.add(
        ids=[metadata.id],
        documents=[metadata.video_description],
        embeddings=[embedding],
        metadatas=[metadata.model_dump()] 
    )
    print("Data successfully added to ChromaDB!")
    return None

def id_exists_in_vector_db(collection, id: str) -> bool:
    result = collection.get(ids=[id])  # Retrieve by ID
    return bool(result['ids'])  # Check if any ID is returned

def hybrid_search(
    collection,
    query: str,
    n_results: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    embedding_model=EMBEDDING_MODEL
) -> Dict[str, Any]:
    """
    Performs a hybrid search on ChromaDB using vector similarity and metadata filters.

    Args:
        collection: The ChromaDB collection.
        query (str): The search query.
        n_results (int): Number of results to retrieve.
        filters (Optional[Dict[str, Any]]): Metadata filters for keyword-based search.

    Returns:
        Dict[str, Any]: Search results from ChromaDB.
    """
    query_vector = embedding_model.encode(query).tolist()
    
    # Perform query with optional filtering
    results = collection.query(
        query_embeddings=[query_vector], 
        n_results=n_results,  
        where=filters if filters else None  # Apply filters if provided
    )

    return results



