from fastapi import FastAPI, Request, File, UploadFile, Form
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

app = FastAPI()

# Mount static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load NLP model
nlp = spacy.load("trained_model")

# Paths to data files
CSV_PATH = "pdfs/statistics"
EXCEL_PATH = "pdfs/report/"
PDF_PATH = "pdfs/LOWs"
TEXT_PATH = "pdfs/pdfs"
LOG_FILE_PATH = "logs/failed_requests.log"

# Load all CSV, Excel, PDF, and Text files into data frames and dictionaries at startup
def load_data():
    data_frames = {}
    pdf_data = {}
    text_data = {}

    # Load CSV files
    for filename in os.listdir(CSV_PATH):
        if filename.endswith(".csv"):
            filepath = os.path.join(CSV_PATH, filename)
            data_frames[filename] = pd.read_csv(filepath)

    # Load Excel files
    for filename in os.listdir(EXCEL_PATH):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
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

# Extract text from PDF files
def extract_text_from_pdf(filepath):
    pdf_reader = PyPDF2.PdfReader(filepath)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Initialize loaded data
data_frames, pdf_data, text_data = load_data()

# Generate a histogram chart for a specific numeric column in the matched row
def generate_chart_for_column(df, column):
    plt.figure(figsize=(8, 4))
    df[column].plot(kind='hist', title=f'Histogram of {column}')
    plt.xlabel(column)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"

# Search within all cells in CSV and Excel files, and provide context with relevant information
def search_and_comment_data(query, data_frames, pdf_data, text_data):
    results = []

    # Search CSV and Excel data
    for filename, df in data_frames.items():
        for index, row in df.iterrows():
            for col in df.columns:
                cell_value = str(row[col])
                if query.lower() in cell_value.lower():
                    comment = {
                        "file": filename,
                        "matched_column": col,
                        "other_columns": {col_name: str(row[col_name]) for col_name in df.columns if col_name != col}
                    }

                    # Check for numeric values in other_columns for additional chart generation
                    numeric_columns = {k: float(v) for k, v in comment["other_columns"].items() if v.replace('.', '', 1).isdigit()}
                    
                    # Generate a chart if numeric data is found in other_columns
                    if numeric_columns:
                        plt.figure(figsize=(8, 4))
                        plt.bar(numeric_columns.keys(), numeric_columns.values())
                        plt.title(f'Contextual Data for {filename} - {col}')
                        plt.xlabel('Attributes')
                        plt.ylabel('Values')
                        buf = BytesIO()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        plt.close()
                        comment["context_chart"] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

                    if pd.api.types.is_numeric_dtype(df[col]):
                        chart = generate_chart_for_column(df, col)
                        comment["chart"] = chart
                        
                    results.append(comment)

    # Search PDF and text data (without charts for these)
    for filename, content in pdf_data.items():
        if query.lower() in content.lower():
            results.append({"file": filename, "matched_text": content[:255] + "..."})

    for filename, content in text_data.items():
        if query.lower() in content.lower():
            results.append({"file": filename, "matched_text": content[:255] + "..."})

    return results if results else [{"error": "No relevant data "}]

# Chatbot response route
@app.post("/get_response", response_class=JSONResponse)
async def get_response(query: str = Form(...)):
    search_results = search_and_comment_data(query, data_frames, pdf_data, text_data)

    if "error" in search_results[0]:
        return {"type": "text", "data": search_results[0]["error"]}
    else:
        return {"type": "contextual_summary", "comments": search_results}

# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# File upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1].lower()
    save_path = ""
    if file_extension == "pdf":
        save_path = os.path.join(PDF_PATH, file.filename)
    elif file_extension in ["xlsx", "xls"]:
        save_path = os.path.join(EXCEL_PATH, file.filename)
    elif file_extension == "csv":
        save_path = os.path.join(CSV_PATH, file.filename)
    elif file_extension == "txt":
        save_path = os.path.join(TEXT_PATH, file.filename)
    else:
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    global data_frames, pdf_data, text_data
    data_frames, pdf_data, text_data = load_data()

    return JSONResponse(content={"message": f"{file.filename} uploaded and dataset updated successfully."})

# Admin dashboard route
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    pdf_files = os.listdir(PDF_PATH)
    excel_files = os.listdir(EXCEL_PATH)
    csv_files = os.listdir(CSV_PATH)
    text_files = os.listdir(TEXT_PATH)
    failed_requests = []
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r") as log_file:
            failed_requests = log_file.readlines()

    return templates.TemplateResponse("admin.html", {
        "request": request,
        "pdf_files": pdf_files,
        "excel_files": excel_files,
        "csv_files": csv_files,
        "text_files": text_files,
        "failed_requests": failed_requests
    })

# Log failed requests
def log_failed_request(query: str):
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(f"{datetime.now()}: {query}\n")
