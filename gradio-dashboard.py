import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
from transformers import pipeline

load_dotenv()

# Load the book dataset
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Load and split text
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Initialize zero-shot classifier
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Sample only 20 books for faster testing
sample_books = books.dropna(subset=["description"]).sample(200, random_state=42)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        final_top_k: int = 8,
) -> pd.DataFrame:
    candidate_labels = [query]
    scores = []

    for _, row in sample_books.iterrows():
        desc = row["description"]
        result = pipe(sequences=desc, candidate_labels=candidate_labels)
        score = result["scores"][0] if result["labels"][0] == query else 0
        scores.append(score)

    sample_books["semantic_score"] = scores
    top_books = sample_books.sort_values(by="semantic_score", ascending=False)

    if category != "All":
        top_books = top_books[top_books["simple_categories"] == category]

    if tone == "Happy":
        top_books = top_books.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        top_books = top_books.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        top_books = top_books.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        top_books = top_books.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        top_books = top_books.sort_values(by="sadness", ascending=False)

    return top_books.head(final_top_k)

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

# Gradio UI
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 🔍 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Enter a book idea or theme:", placeholder="e.g., A story about resilience")
        category_dropdown = gr.Dropdown(choices=categories, label="Select category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select emotional tone:", value="All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## 📚 Your Recommendations")
    output = gr.Gallery(label="Books", columns=4, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()
