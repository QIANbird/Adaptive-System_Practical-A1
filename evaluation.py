"""Task 1–3 共用的 top-k 类别一致性评估。

本模块会对指定类别中的每篇文章进行相似文章检索，排除文章自身后，
检查前 top_k 篇结果中有多少篇仍属于该类别，并汇总 Precision@k。
"""


def evaluate_quality(articles, vectors, similarity_index,
                     section="Food & Drink", top_k=10):
    """计算指定类别文章的平均 Precision@k。

    参数：
        articles: 文章记录列表。每条记录至少需要包含 article_section 字段。
        vectors: 与 articles 一一对应的文章向量列表。
        similarity_index: 可通过查询向量取回它与语料库各向量相似度的索引。
        section: 要评估的目标类别，默认是 "Food & Drink"。
        top_k: 每篇目标文章要检查的相似文章数量，必须大于 0。

    返回：
        包含语料规模、目标文章数量、相关结果总数和平均 Precision@k 的字典。

    注意：
        检索结果会排除查询文章自身；相似度相同时保持原有 CSV 行顺序。
        使用清洗后的文章列表和对应向量，分母按清洗后的目标类别文章数计算。
        函数本身不统计耗时，耗时由调用方记录。
    """
    # 每篇文章都必须有一个对应向量；否则索引位置无法正确对应。
    if len(articles) != len(vectors):
        raise ValueError("Article and vector counts must match")

    # 查询文章自身会被排除，因此语料至少要比 top_k 多一篇。
    # 同时拒绝 top_k 小于 1 的无效设置。
    if top_k < 1 or len(articles) <= top_k:
        raise ValueError("The corpus must contain at least top_k + 1 articles")

    # 找出所有属于目标类别的文章索引。
    target_ids = [i for i, article in enumerate(articles)
                  if article["article_section"] == section]
    if not target_ids:
        raise ValueError(f"No articles found for section: {section}")

    # 统计所有目标文章的前 top_k 个相似结果中，属于目标类别的文章总数。
    total_goods = 0
    for article_id in target_ids:
        # 用当前文章的向量查询相似度索引。
        # sims 中每个分数的位置对应 articles / vectors 中的文章索引。
        sims = similarity_index[vectors[article_id]]

        # 将相似度分数与文章索引配对，并过滤掉文章自身。
        # 按相似度从高到低排序；相似度相同时，Python 的稳定排序会保留
        # enumerate 产生的原始行顺序（即 CSV 中的顺序）。
        ranked = sorted(
            ((i, score) for i, score in enumerate(sims) if i != article_id),
            key=lambda item: -item[1],
        )

        # 只检查排序后的前 top_k 篇文章，统计其中与查询文章同类别的数量。
        total_goods += sum(
            articles[i]["article_section"] == section
            for i, _ in ranked[:top_k]
        )

    # 汇总评估结果。
    # ratio_quality 即平均 Precision@k：
    # 所有目标文章的相关结果总数 / (目标文章数 * top_k)。
    return {
        "section": section,
        "corpus_size": len(articles),
        "num_target_articles": len(target_ids),
        "top_k": top_k,
        "exclude_self": True,
        "total_goods": total_goods,
        "ratio_quality": total_goods / (len(target_ids) * top_k),
    }
