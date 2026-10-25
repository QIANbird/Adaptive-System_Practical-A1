"""Task 1 / Task 3：从 tfidf_example.py 拓展的 TF-IDF 实验入口。"""

from time import perf_counter

# 尽早开始计时，覆盖后续依赖导入、读取、预处理与索引构建。
PROGRAM_START = perf_counter()

import argparse
import json

from gensim import models, similarities

from evaluation import evaluate_quality
from preprocessing import DEFAULT_DATA_PATH, build_corpus, load_articles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--section", default="Food & Drink")
    parser.add_argument("--limit", type=int, help="Smoke check only: first N rows")
    args = parser.parse_args()

    articles = load_articles(args.data, args.limit)
    dictionary, model_bow = build_corpus(articles)
    tfidf = models.TfidfModel(model_bow)
    tfidf_vectors = list(tfidf[model_bow])
    matrix_tfidf = similarities.MatrixSimilarity(
        tfidf_vectors, num_features=len(dictionary)
    )
    model_seconds = perf_counter() - PROGRAM_START

    evaluation_start = perf_counter()
    result = evaluate_quality(articles, tfidf_vectors, matrix_tfidf, args.section)
    evaluation_seconds = perf_counter() - evaluation_start

    result.update({
        "model": "tfidf",
        "sample_limit": args.limit,
        "model_creation_seconds": model_seconds,
        "evaluation_seconds": evaluation_seconds,
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
