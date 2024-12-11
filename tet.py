from flask import Flask, request, jsonify, render_template
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
from datetime import datetime
import spacy
from transformers import pipeline
import math
from flask_cors import CORS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import requests
import time

SCRAPED_URLS = set()  # To keep track of already scraped URLs
MAX_DEPTH = 1  # Set a maximum depth to avoid excessive recursion

app = Flask(__name__)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load a pretrained transformer pipeline for question answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# File paths
RECORDED_URLS_FILE = "recorded_urls.json"
USERS_FILE = "users.json"
LOGS_FILE = "logs.json"

# Initialize files if they don't exist
def initialize_file(file_path, default_data=[]):
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(default_data, file)
initialize_file(RECORDED_URLS_FILE)
initialize_file(USERS_FILE)
initialize_file(LOGS_FILE)

# Utility functions for file operations
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Logging admin actions
def log_action(action, details):
    logs = load_json(LOGS_FILE)
    logs.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "details": details
    })
    save_json(logs, LOGS_FILE)

# Web scraping logic
def fetch_and_save_webpage_data(url, depth=0):
    """Fetch and scrape data from a webpage and its links recursively."""
    if url in SCRAPED_URLS:
        print(f"Skipping already scraped URL: {url}")
        return

    if depth > MAX_DEPTH:
        print(f"Max depth reached for URL: {url}")
        return

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Mark this URL as scraped
        SCRAPED_URLS.add(url)
        print(f"Scraping URL: {url}")

        soup = BeautifulSoup(response.text, "lxml")

        # Extract headings, paragraphs, and metadata
        headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        title = soup.title.string if soup.title else ""
        
        # Extract all links on the page
        links = [urljoin(url, a['href']) for a in soup.find_all("a", href=True)]

        # Clean and filter links (only HTTP/HTTPS links)
        links = [link for link in links if urlparse(link).scheme in ["http", "https"]]

        # Save the data
        data = {
            "url": url,
            "title": title,
            "headings": headings,
            "paragraphs": paragraphs,
            "links": links
        }

        save_data_to_json(data)

        # Recursively scrape each found link
        for link in links:
            time.sleep(1)  # Be polite and avoid overwhelming the server
            fetch_and_save_webpage_data(link, depth=depth + 1)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def save_data_to_json(new_data):
    """Save new data to the recorded URLs JSON file."""
    try:
        with open("recorded_urls.json", "r", encoding="utf-8") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(new_data)

    with open("recorded_urls.json", "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def save_data_to_json(new_data):
    data = load_json(RECORDED_URLS_FILE)
    data.append(new_data)
    save_json(data, RECORDED_URLS_FILE)

# Flask routes
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/record_url', methods=['POST'])
def record_url():
    """Record and scrape data from a URL."""
    try:
        url = request.json.get('url')
        print(f"Received URL: {url}")
        if not url:
            return jsonify({"error": "URL is required"}), 400
        data = fetch_and_save_webpage_data(url)
        if "error" in data:
            return jsonify({"error": data["error"]}), 400
        return jsonify({"message": "URL recorded and scraped successfully!"}), 200
    except Exception as e:
        print(f"Exception: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

@app.route('/view_urls', methods=['GET'])
def view_urls():
    return jsonify(load_json(RECORDED_URLS_FILE))

@app.route('/admin/add_user', methods=['POST'])
def add_user():
    name = request.json.get('name')
    email = request.json.get('email')
    whatsapp_contact = request.json.get('whatsapp_contact')
    role = request.json.get('role', 'user')
    if not all([name, email, whatsapp_contact]):
        return jsonify({"error": "Name, email, and WhatsApp contact are required"}), 400

    users = load_json(USERS_FILE)
    if any(user['email'] == email for user in users):
        return jsonify({"error": "User with this email already exists!"}), 400

    new_user = {
        "name": name,
        "email": email,
        "whatsapp_contact": whatsapp_contact,
        "role": role,
        "created_at": datetime.now().isoformat()
    }
    users.append(new_user)
    save_json(users, USERS_FILE)
    log_action("Add User", f"Added user {name} with email {email} and role {role}.")
    return jsonify({"message": "User added successfully!", "user": new_user}), 201
@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint with transformer-based NLP."""
    query = request.json.get('query', '').strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    data = load_json(RECORDED_URLS_FILE)
    if not data:
        return jsonify({"error": "No recorded data available."}), 400

    answers = process_query_with_transformers(query, data)

    if not answers:
        return jsonify([{"message": "No relevant data found."}])

    return jsonify(answers)


# def process_query_with_transformers(query, data):
#     """Process user query using transformers for semantic understanding."""
#     results = []

#     for entry in data:
#         content = " ".join(entry.get("headings", []) + entry.get("paragraphs", []))
#         if not content.strip():
#             print(f"No valid content for URL: {entry['url']}")
#             continue

#         # Debug the content being processed
#         print(f"Processing URL: {entry['url']}")
#         print(f"Content snippet: {content[:200]}...")

#         try:
#             response = qa_pipeline(question=query, context=content)
#             print(f"Pipeline Response: {response}")

#             answer = response.get("answer", "No answer found")
#             score = response.get("score", 0.0)

#             # Handle NaN scores
#             if math.isnan(score):
#                 score = 0.0

#             results.append({
#                 "url": entry["url"],
#                 "answer": answer,
#                 "confidence": f"{score:.2f}"
#             })
#         except Exception as e:
#             print(f"Error processing entry for URL {entry['url']}: {e}")

#     return sorted(results, key=lambda x: float(x["confidence"]), reverse=True)
MIN_CONFIDENCE_THRESHOLD = 0.1

def process_query_with_transformers(query, data):
    results = []

    for entry in data:
        content = " ".join(entry.get("headings", []) + entry.get("paragraphs", []))
        content = content[:2000]

        if not content.strip():
            continue

        try:
            response = qa_pipeline(question=query, context=content)
            score = response.get("score", 0.0)

            # Only keep results above a minimum confidence threshold
            if score >= MIN_CONFIDENCE_THRESHOLD:
                results.append({
                    "url": entry["url"],
                    "answer": response["answer"],
                    "confidence": f"{score:.2f}"
                })
        except Exception as e:
            print(f"Error processing entry for URL {entry['url']}: {e}")

    return sorted(results, key=lambda x: float(x["confidence"]), reverse=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
