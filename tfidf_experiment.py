"""Entry point for the TF-IDF experiment."""

from time import perf_counter
import argparse
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

    # Preprocess the data
    model_start = perf_counter()
    articles = load_articles(args.data)
    dictionary, model_bow = build_corpus(articles)

    # Build the TF-IDF model
    tfidf = models.TfidfModel(model_bow)
    tfidf_vectors = list(tfidf[model_bow])
    matrix_tfidf = similarities.MatrixSimilarity(
        tfidf_vectors, num_features=len(dictionary)
    )
    model_seconds = perf_counter() - model_start

    # Query similarities and evaluate the model.
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
