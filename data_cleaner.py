import json
import pandas as pd

# Load the JSON data
with open('scraped_page_data.json', 'r') as file:
    data = json.load(file)

# Display the main keys
print("Main Keys in JSON:", data.keys())

# Extract the main components
url = data.get("url", "")
headings = data.get("headings", [])
paragraphs = data.get("paragraphs", [])
links = data.get("links", [])
scraped_links = data.get("scraped_links", [])

# Create DataFrames for analysis
df_paragraphs = pd.DataFrame(paragraphs, columns=["Paragraph"])
df_links = pd.DataFrame(links, columns=["Link"])
df_scraped_links = pd.DataFrame(scraped_links)

# Display a summary of each DataFrame
print("\nParagraphs DataFrame:")
print(df_paragraphs.head())

print("\nLinks DataFrame:")
print(df_links.head())

print("\nScraped Links DataFrame:")
print(df_scraped_links.head())
