# Shared data processing. Reference: vectormodel_stem_example.py
# This module reads the news CSV file, cleans article descriptions, and builds a Gensim dictionary and Bag of Words (BoW) vectors for use by later model and evaluation scripts.


import csv #Used to read the news data file
from pathlib import Path #Used to handle file paths in a cross-platform way
from gensim import corpora #Provides text dictionary and bag-of-words vector functionality.
from nltk import PorterStemmer #Provides the Porter stemmer and English stopword functionality
from nltk.corpus import stopwords


# Data file location
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "news2.csv"

# Read the news CSV file, remove articles with missing or duplicate descriptions, and return the list of valid articles.
def load_articles(data_path=DEFAULT_DATA_PATH, limit=None):

    with Path(data_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"headline", "tags", "article_section", "description"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain these columns: {sorted(required)}")

        articles = []
        seen_descriptions = set()
        rows_read = 0

        # Clean and deduplicate using article descriptions. Convert missing values to empty strings; keep only the first article with each non-empty description.
        for rows_read, row in enumerate(reader, start=1):

            article = {key: (row.get(key) or "").strip() for key in required}
            description = article["description"]

            if description and description not in seen_descriptions:
                seen_descriptions.add(description)
                articles.append(article)

            if limit is not None and rows_read >= limit:
                break

    if rows_read == 0:
        raise ValueError("The dataset contains no articles")
    
    if not articles:
        raise ValueError("No articles remain after removing empty and duplicate descriptions")
    return articles

# Clean article descriptions and convert them into lists of tokens
# Convert to lowercase → split on whitespace → remove English stopwords → apply Porter stemming
def preprocess_descriptions(articles):

    # Stemmer
    porter = PorterStemmer()

    # Create a set of stopwords
    try:
        stoplist = set(stopwords.words("english"))
    except LookupError as error:
        raise RuntimeError(
            "Missing NLTK stopwords"
        ) from error

    return [
        [porter.stem(word) for word in article["description"].lower().split()
         if word not in stoplist]
        for article in articles
    ]

# Build a dictionary and bag-of-words vectors from article descriptions
def build_corpus(articles):

    texts = preprocess_descriptions(articles)

    dictionary = corpora.Dictionary(texts)
    if not dictionary:
        raise ValueError("The dataset contains no usable tokens")

    model_bow = [dictionary.doc2bow(text) for text in texts]
    
    return dictionary, model_bow
