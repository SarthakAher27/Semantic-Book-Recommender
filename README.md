# 📚 Semantic Book Recommender using LLMs

This project is a full-stack book recommendation system powered by open-source LLM tools. It allows users to search for books using natural language queries and get recommendations based on semantic similarity, emotional tone, and category (Fiction/Nonfiction).

## 🚀 Features

- 🔍 **Natural language search**: Type something like `"a suspenseful journey of self-discovery"` and get semantically relevant books.
- 📊 **Zero-shot classification**: Categorizes books as **Fiction** or **Nonfiction** using `facebook/bart-large-mnli`.
- 🎭 **Emotion-based sorting**: Ranks books based on emotional tones like **joy**, **fear**, **anger**, **sadness**, and **surprise**.
- 🌐 **Interactive UI**: Built with **Gradio** for a smooth user experience.
- ✅ **Fully open-source**: No paid APIs. All models used are from Hugging Face.

## 🧠 Tech Stack

| Component          | Tool/Model                                 |
|-------------------|---------------------------------------------|
| Embeddings         | `sentence-transformers/all-MiniLM-L6-v2`    |
| Vector Database    | ChromaDB                                    |
| Classification     | `facebook/bart-large-mnli`                  |
| UI Framework       | Gradio                                      |
| Core Libraries     | Python, LangChain, Transformers, Pandas     |

## 📁 Project Structure
📦 book-recommender/
├── data-exploration.ipynb # Text cleaning & preprocessing
├── vector-search.ipynb # Embedding + Vector DB setup
├── text-classification.ipynb # Zero-shot classification
├── sentiment-analysis.ipynb # Emotion scoring
├── gradio-dashboard.py # Frontend app with Gradio
├── tagged_description.txt # Raw book descriptions
├── books_with_emotions.csv # Final dataset
├── requirements.txt # Python dependencies
└── .env # Environment variables
