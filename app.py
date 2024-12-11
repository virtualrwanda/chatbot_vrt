from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import shutil
import PyPDF2
from datetime import datetime
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import spacy
import pandas as pd
import PyPDF2
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Mount static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load NLP model
try:
    nlp = spacy.load("trained_model")  # Replace with your specific model path or name
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading NLP model: {e}")

# Paths to data files
CSV_PATH = "pdfs/statistics"
EXCEL_PATH = "pdfs/report"
PDF_PATH = "pdfs/LOWs"
TEXT_PATH = "pdfs/pdfs"
LOG_FILE_PATH = "logs/failed_requests.log"

# Ensure directories exist
os.makedirs(CSV_PATH, exist_ok=True)
os.makedirs(EXCEL_PATH, exist_ok=True)
os.makedirs(PDF_PATH, exist_ok=True)
os.makedirs(TEXT_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# Load data at startup
def load_data():
    data_frames, pdf_data, text_data = {}, {}, {}

    # Load CSV files
    for filename in os.listdir(CSV_PATH):
        if filename.endswith(".csv"):
            filepath = os.path.join(CSV_PATH, filename)
            data_frames[filename] = pd.read_csv(filepath)

    # Load Excel files
    for filename in os.listdir(EXCEL_PATH):
        if filename.endswith((".xlsx", ".xls")):
            filepath = os.path.join(EXCEL_PATH, filename)
            data_frames[filename] = pd.read_excel(filepath)

    # Load PDF files
    for filename in os.listdir(PDF_PATH):
        if filename.endswith(".pdf"):
            filepath = os.path.join(PDF_PATH, filename)
            pdf_data[filename] = extract_text_from_pdf(filepath)

    # Load text files
    for filename in os.listdir(TEXT_PATH):
        if filename.endswith(".txt"):
            filepath = os.path.join(TEXT_PATH, filename)
            with open(filepath, "r") as file:
                text_data[filename] = file.read()

    return data_frames, pdf_data, text_data

def extract_text_from_pdf(filepath):
    text = ""
    try:
        with open(filepath, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        log_failed_request(f"PDF extraction failed: {filepath} - {e}")
    return text

# Initial data load
data_frames, pdf_data, text_data = load_data()

# Generate an image from the numeric data for visualization
# Assuming these are defined at the top of your script
data_frames, pdf_data, text_data = load_data()

def search_and_comment_data(query, data_frames, pdf_data, text_data):
    results = []

    # Extract keywords using NLP
    doc = nlp(query)
    search_terms = [ent.text for ent in doc.ents] + [
        token.text for token in doc if token.is_alpha and not token.is_stop
    ]

    # Search in CSV and Excel files
    for filename, df in data_frames.items():
        for _, row in df.iterrows():
            row_content = {col: str(row[col]) for col in df.columns}
            numeric_data = {col: float(row[col]) for col in df.columns if str(row[col]).replace('.', '', 1).isdigit()}
            
            chart_image = generate_example_image(numeric_data) if numeric_data else None

            results.append({
                "row_content": row_content,
                "chart_image": chart_image,
                "comment": f"Relevant data identified in {filename}."
            })

    # Search in PDFs and text files
    for filename, content in pdf_data.items():
        if any(term.lower() in content.lower() for term in search_terms):
            results.append({
                "row_content": content[:255] + "...",
                "chart_image": None,
                "comment": f"Relevant content found in PDF: {filename}."
            })

    for filename, content in text_data.items():
        if any(term.lower() in content.lower() for term in search_terms):
            results.append({
                "row_content": content[:255] + "...",
                "chart_image": None,
                "comment": f"Relevant content found in text file: {filename}."
            })

    return results if results else [{"error": "No relevant data found."}]

# Chatbot response route
# Sample data for the chatbot to simulate responses
@app.post("/get_response", response_class=JSONResponse)
async def get_response(query: str = Form(...)):
    if not query:
        return {"type": "text", "data": "Please enter a valid question."}

    # Simulated chatbot logic with integrated search function
    try:
        # Check for greetings
        if "hello" in query.lower():
            return {"type": "text", "data": "Hi there! How can I assist you today?"}

        # Check for specific keyword "data"
        elif "data" in query.lower():
            # Example response with simulated data
            return {
                "type": "contextual_summary",
                "comments": [
                    {
                        "row_content": "Sample data row content",
                        "chart_image": None,
                        "comment": "Here's some relevant data."
                    }
                ]
            }

        # Perform data search with the `search_and_comment_data` function
        else:
            search_results = search_and_comment_data(query, data_frames, pdf_data, text_data)
            if "error" in search_results[0]:
                return {"type": "text", "data": search_results[0]["error"]}
            else:
                return {"type": "contextual_summary", "comments": search_results}

    except Exception as e:
        # Handle unexpected errors gracefully
        return {"type": "text", "data": f"An error occurred: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# File upload route
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1].lower()
    save_paths = {
        "pdf": PDF_PATH,
        "xlsx": EXCEL_PATH,
        "xls": EXCEL_PATH,
        "csv": CSV_PATH,
        "txt": TEXT_PATH
    }

    save_path = save_paths.get(file_extension)
    if not save_path:
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    full_path = os.path.join(save_path, file.filename)
    with open(full_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    global data_frames, pdf_data, text_data
    data_frames, pdf_data, text_data = load_data()

    return JSONResponse(content={"message": f"{file.filename} uploaded and dataset updated successfully."})

# Log failed request
def log_failed_request(query: str):
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"{datetime.now()}: {query}\n")
        
def is_plot_request(query: str) -> bool:
    keywords = ["plot", "chart", "graph", "statistics", "trend"]
    return any(keyword in query.lower() for keyword in keywords)

# Function to generate chart from data
def generate_chart(df: pd.DataFrame, columns: List[str], chart_type: str = "line") -> str:
    try:
        plt.figure(figsize=(10, 6))
        if chart_type == "line":
            df[columns].plot(kind="line")
        elif chart_type == "bar":
            df[columns].plot(kind="bar")
        elif chart_type == "hist":
            df[columns].plot(kind="hist")

        plt.title(f"{chart_type.capitalize()} Chart for {', '.join(columns)}")
        plt.xlabel("Index")
        plt.ylabel("Values")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        return f"Error generating chart: {e}"
# Function to search numeric data in dataframes and create a plot
def search_and_plot_data(query: str, data_frames: Dict[str, pd.DataFrame]) -> str:
    columns = []
    chart_type = "line"

    for filename, df in data_frames.items():
        for col in df.columns:
            if col.lower() in query.lower():
                columns.append(col)

    if not columns:
        return "No relevant numeric data found for generating a plot."

    df = data_frames[next(iter(data_frames))]
    return generate_chart(df, columns, chart_type=chart_type)

# Function to search PDF data
def search_pdf_data(query: str, pdf_data: Dict[str, str]) -> List[str]:
    matches = []
    for filename, text in pdf_data.items():
        if query.lower() in text.lower():
            matches.append(f"{filename}: {text[:500]}...")
    return matches

# Function to search data in Excel and CSV files
def search_data_frames(query: str, data_frames: Dict[str, pd.DataFrame]) -> List[str]:
    results = []
    for filename, df in data_frames.items():
        matches = df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)
        if matches.any():
            results.append(f"{filename}:\n{df[matches].to_string(index=False)}")
    return results

# Summarize collected information into an understandable answer
def generate_detailed_response(query: str, pdf_data: Dict[str, str], data_frames: Dict[str, pd.DataFrame]) -> str:
    pdf_results = search_pdf_data(query, pdf_data)
    data_frame_results = search_data_frames(query, data_frames)
    
    all_results = pdf_results + data_frame_results
    if not all_results:
        return "No relevant information found in the data bank."

    combined_text = " ".join(all_results)
    sentences = combined_text.split(". ")
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)
    query_vector = vectorizer.transform([query])
    
    scores = cosine_similarity(query_vector, vectors)[0]
    ranked_sentences = [sent for _, sent in sorted(zip(scores, sentences), reverse=True)]
    
    response = " ".join(ranked_sentences[:4])
    return response