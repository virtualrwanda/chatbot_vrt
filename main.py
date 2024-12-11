from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import pandas as pd
import pdfplumber
import fitz  # PyMuPDF
import json
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

# Global storage for data and embeddings
data_store = []
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, efficient model for embeddings

# ----- Data Ingestion Functions -----

def process_pdf(file_path: str) -> Dict:
    """Extract text from a PDF file."""
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return {"type": "pdf", "content": text, "file_path": file_path}

def process_excel(file_path: str) -> Dict:
    """Read Excel data into a dictionary format."""
    df = pd.read_excel(file_path)
    return {"type": "excel", "content": df.to_dict(), "file_path": file_path}

def process_text(file_path: str) -> Dict:
    """Read and return text file content."""
    with open(file_path, 'r') as f:
        text = f.read()
    return {"type": "text", "content": text, "file_path": file_path}

def process_json(file_path: str) -> Dict:
    """Read and return JSON file content."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {"type": "json", "content": data, "file_path": file_path}

def process_image(file_path: str) -> Dict:
    """Use OCR to extract text from an image."""
    text = pytesseract.image_to_string(Image.open(file_path))
    return {"type": "image", "content": text, "file_path": file_path}

# Load all data on startup
@app.on_event("startup")
def load_data():
    directory = "path_to_data"  # Specify your data directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            data_store.append(process_pdf(file_path))
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            data_store.append(process_excel(file_path))
        elif filename.endswith(".txt"):
            data_store.append(process_text(file_path))
        elif filename.endswith(".json"):
            data_store.append(process_json(file_path))
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            data_store.append(process_image(file_path))

# ----- Embedding and Search Functions -----

def generate_embedding(text: str) -> np.ndarray:
    """Generate embeddings for the text using a sentence transformer."""
    return embedding_model.encode(text)

def search_documents(query: str, top_n: int = 5) -> List[Dict]:
    """Search for documents most similar to the query."""
    query_embedding = generate_embedding(query)
    results = []
    
    for document in data_store:
        content_embedding = generate_embedding(document["content"]) if document["type"] != "excel" else None
        similarity_score = np.dot(query_embedding, content_embedding) if content_embedding is not None else 0
        results.append({"document": document, "score": similarity_score})
    
    # Sort results by similarity score in descending order and return top N
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]
    return sorted_results

# ----- API Models -----

class QueryRequest(BaseModel):
    query: str

class DocumentResponse(BaseModel):
    document_type: str
    content: str
    file_path: str
    score: float

# ----- API Endpoints -----

@app.post("/search", response_model=List[DocumentResponse])
def search_data(query_request: QueryRequest):
    """Endpoint to search the databank for relevant documents."""
    query = query_request.query
    results = search_documents(query)
    
    if not results:
        raise HTTPException(status_code=404, detail="No relevant data found")
    
    return [{"document_type": res["document"]["type"],
             "content": res["document"]["content"],
             "file_path": res["document"]["file_path"],
             "score": res["score"]} for res in results]

# ----- Run the API -----

# You can run this FastAPI app using Uvicorn from the command line as follows:
# uvicorn main:app --reload
