"""Task 2 / Task 3：从 lda_example.py 拓展的 LDA 实验入口。"""

from time import perf_counter

import argparse
import json

from gensim import models, similarities

from evaluation import evaluate_quality
from preprocessing import DEFAULT_DATA_PATH, build_corpus, load_articles


NUM_TOPICS = 30
PASSES = 2
RANDOM_STATE = 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--section", default="Food & Drink")
    args = parser.parse_args()

    # 与 TF-IDF 一致：参数解析完成后、读取数据前开始计时。
    # 排除解释器启动、库导入和参数解析；实际耗时仍受硬件与系统负载影响。
    model_start = perf_counter()

    articles = load_articles(args.data)
    dictionary, model_bow = build_corpus(articles)
    lda = models.LdaModel(
        model_bow, id2word=dictionary, num_topics=NUM_TOPICS,
        passes=PASSES, random_state=RANDOM_STATE,
    )
    # 每篇文章仅转换一次；查询复用向量，避免重复推断。
    lda_vectors = [lda[vector] for vector in model_bow]
    matrix_lda = similarities.MatrixSimilarity(lda_vectors, num_features=NUM_TOPICS)
    # 索引构建完成后结束第一段计时，包含数据处理与模型构建。
    model_seconds = perf_counter() - model_start

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
