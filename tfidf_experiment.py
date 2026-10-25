"""Entry point for the TF-IDF experiment."""

from time import perf_counter
import argparse  # Parse command-line arguments passed to the script.
import json
from gensim import models, similarities
from evaluation import evaluate_quality
from preprocessing import DEFAULT_DATA_PATH, build_corpus, load_articles


def main():
    # Configure command-line arguments: data specifies the dataset path,
    # and section specifies the category to evaluate.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--section", default="Food & Drink")
    args = parser.parse_args()

    # Start timing model creation.
    model_start = perf_counter()

    # Load the raw data and preprocess it.
    articles = load_articles(args.data)
    dictionary, model_bow = build_corpus(articles)

    # Convert each article's BoW vector into a TF-IDF vector.
    tfidf = models.TfidfModel(model_bow)
    tfidf_vectors = list(tfidf[model_bow])

    matrix_tfidf = similarities.MatrixSimilarity(
        tfidf_vectors, num_features=len(dictionary)
    )
    # Stop timing data processing and model creation.
    model_seconds = perf_counter() - model_start

    # Start timing model evaluation.
    evaluation_start = perf_counter()
    # Evaluate model performance for the specified category.
    result = evaluate_quality(articles, tfidf_vectors, matrix_tfidf, args.section)
    evaluation_seconds = perf_counter() - evaluation_start

    # Add experiment metadata and timing information to the JSON report.
    result.update({
        "model": "tfidf",
        "model_creation_seconds": model_seconds,
        "evaluation_seconds": evaluation_seconds,
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
