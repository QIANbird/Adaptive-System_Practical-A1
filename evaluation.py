"""Task 1–3 共用的 top-k 类别一致性评估。"""


def evaluate_quality(articles, vectors, similarity_index,
                     section="Food & Drink", top_k=10):
    """在全语料库中检索，排除查询文章自身，计算平均 Precision@k。

    同分时沿用 CSV 行顺序；保留空描述与重复文章。
    返回计数和指标，耗时由实验入口在调用前后记录。
    """
    if len(articles) != len(vectors):
        raise ValueError("Article and vector counts must match")
    if top_k < 1 or len(articles) <= top_k:
        raise ValueError("The corpus must contain at least top_k + 1 articles")
    target_ids = [i for i, article in enumerate(articles)
                  if article["article_section"] == section]
    if not target_ids:
        raise ValueError(f"No articles found for section: {section}")

    total_goods = 0
    for article_id in target_ids:
        sims = similarity_index[vectors[article_id]]
        # 与课堂示例一致：enumerate 后按相似度降序排序。
        ranked = sorted(
            ((i, score) for i, score in enumerate(sims) if i != article_id),
            key=lambda item: -item[1],
        )
        total_goods += sum(
            articles[i]["article_section"] == section
            for i, _ in ranked[:top_k]
        )

    return {
        "section": section,
        "corpus_size": len(articles),
        "num_target_articles": len(target_ids),
        "top_k": top_k,
        "exclude_self": True,
        "total_goods": total_goods,
        "ratio_quality": total_goods / (len(target_ids) * top_k),
    }
