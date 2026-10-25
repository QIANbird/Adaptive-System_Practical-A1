"""Entry point for the LDA experiment."""

from time import perf_counter
import argparse
import json
from gensim import models, similarities
from evaluation import evaluate_quality
from preprocessing import DEFAULT_DATA_PATH, build_corpus, load_articles

# Constants for LDA model configuration
NUM_TOPICS = 30
PASSES = 2
RANDOM_STATE = 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--section", default="Food & Drink")
    args = parser.parse_args()

    # Preprocess the data.
    model_start = perf_counter()

    articles = load_articles(args.data)
    dictionary, model_bow = build_corpus(articles)

    # Build the LDA model.
    lda = models.LdaModel(
        model_bow, id2word=dictionary, num_topics=NUM_TOPICS,
        passes=PASSES, random_state=RANDOM_STATE,
    )
    lda_vectors = [lda[vector] for vector in model_bow]
    matrix_lda = similarities.MatrixSimilarity(lda_vectors, num_features=NUM_TOPICS)
    model_seconds = perf_counter() - model_start

    # Query similarities and evaluate the model.
    evaluation_start = perf_counter()

    result = evaluate_quality(articles, lda_vectors, matrix_lda, args.section)
    evaluation_seconds = perf_counter() - evaluation_start

    result.update({
        "model": "lda",
        "num_topics": NUM_TOPICS,
        "passes": PASSES,
        "random_state": RANDOM_STATE,
        "model_creation_seconds": model_seconds,
        "evaluation_seconds": evaluation_seconds,
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
