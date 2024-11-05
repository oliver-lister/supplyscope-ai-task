# SupplyScope Reverse Image Recall Search

## Description

This project is a reverse image search tool designed to identify product recall
records in Australia based on visual similarity. By using OpenAI's CLIP model,
the tool compares input images against a dataset of product images to find the
closest matches, providing a confidence score for each match. The project
includes a dataset of recall records with image URLs, and it leverages cosine
similarity to compute and rank image similarities.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/oliver-lister/supplyscope-ai-task.git
   cd supplyscope-ai-task
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Execute Script
I have already run the script across the dataset and stored features inside a pickle file.

```bash
python reverse-image-search.py
```
