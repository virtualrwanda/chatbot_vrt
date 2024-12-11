import json
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.express as px

# ------------------------------
# Step 1: Load the JSON Data
# ------------------------------

with open('scraped_page_data.json', 'r') as file:
    data = json.load(file)

# ------------------------------
# Step 2: Extract Main Components
# ------------------------------

url = data.get("url", "")
headings = data.get("headings", [])
paragraphs = data.get("paragraphs", [])
links = data.get("links", [])
scraped_links = data.get("scraped_links", [])

print(f"Main URL: {url}")
print(f"\nNumber of Paragraphs: {len(paragraphs)}")
print(f"Number of Links: {len(links)}")
print(f"Number of Scraped Links: {len(scraped_links)}")

# ------------------------------
# Step 3: Create DataFrames
# ------------------------------

df_paragraphs = pd.DataFrame(paragraphs, columns=["Paragraph"])
df_links = pd.DataFrame(links, columns=["Link"])
df_scraped_links = pd.DataFrame(scraped_links)

print("\nParagraphs DataFrame:")
print(df_paragraphs.head())

print("\nLinks DataFrame:")
print(df_links.head())

print("\nScraped Links DataFrame:")
print(df_scraped_links.head())

# ------------------------------
# Step 4: Clean the Data
# ------------------------------

# Remove duplicates
df_paragraphs.drop_duplicates(inplace=True)
df_links.drop_duplicates(inplace=True)

# Clean scraped_links DataFrame by removing duplicate paragraphs within each entry
df_scraped_links['paragraphs'] = df_scraped_links['paragraphs'].apply(lambda x: list(set(x)) if isinstance(x, list) else [])

print("\nCleaned Paragraphs DataFrame:")
print(df_paragraphs.head())

print("\nCleaned Links DataFrame:")
print(df_links.head())

# ------------------------------
# Step 5: Save the Cleaned Data
# ------------------------------

df_paragraphs.to_csv('cleaned_paragraphs.csv', index=False)
df_links.to_csv('cleaned_links.csv', index=False)
df_scraped_links.to_json('cleaned_scraped_links.json', indent=2)

print("\nCleaned data has been saved to 'cleaned_paragraphs.csv', 'cleaned_links.csv', and 'cleaned_scraped_links.json'.")

# ------------------------------
# Step 6: Analyze the Data
# ------------------------------

# Define keywords
keywords = list(set([
    'women', 'gender', 'justice', 'laws', 'family', 'opportunity',
    'rights', 'equality', 'empowerment', 'violence', 'protection',
    'education', 'health', 'employment', 'leadership', 'Rwanda',
    'economic', 'finance', 'growth', 'development', 'men'
]))

# Count keyword occurrences in paragraphs
keyword_counts = {keyword: df_paragraphs['Paragraph'].str.contains(keyword, case=False).sum() for keyword in keywords}

print("\nKeyword Frequency in Paragraphs:")
for keyword, count in keyword_counts.items():
    print(f"'{keyword}' appears {count} times.")

# Plot the keyword frequencies
plt.figure(figsize=(12, 7))
plt.bar(keyword_counts.keys(), keyword_counts.values(), color='lightcoral')
plt.title('Keyword Frequency in Paragraphs (Including Gender Terms)')
plt.xlabel('Keywords')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Filter paragraphs containing relevant keywords
relevant_paragraphs = df_paragraphs[df_paragraphs['Paragraph'].str.contains('|'.join(keywords), case=False)]
print("\nParagraphs with Relevant Keywords:")
print(relevant_paragraphs)

# Sentiment analysis
df_paragraphs['Sentiment'] = df_paragraphs['Paragraph'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Categorize sentiment
def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    return 'Neutral'

df_paragraphs['Sentiment Category'] = df_paragraphs['Sentiment'].apply(categorize_sentiment)
print("\nParagraphs with Sentiment Scores:")
print(df_paragraphs[['Paragraph', 'Sentiment', 'Sentiment Category']])

# Generate word cloud
all_text = ' '.join(relevant_paragraphs['Paragraph'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Relevant Paragraphs')
plt.show()

# Plot sentiment distribution
sentiment_counts = df_paragraphs['Sentiment Category'].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution in Relevant Paragraphs')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Create an interactive bar chart for keyword frequency
fig = px.bar(
    x=keyword_counts.keys(),
    y=keyword_counts.values(),
    labels={'x': 'Keywords', 'y': 'Count'},
    title='Keyword Frequency in Paragraphs (Interactive)'
)
fig.update_layout(xaxis_tickangle=-45)
fig.show()
