"""For each article in a specified category, retrieve similar articles,
exclude the query article, and calculate ratio_quality based on how many
of the top_k results belong to the same category.
"""

# Calculate the average ratio_quality for articles in the specified category.
def evaluate_quality(articles, vectors, similarity_index,
                     section="Food & Drink", top_k=10):

    # Validate the input arguments.
    if len(articles) != len(vectors):
        raise ValueError("Article and vector counts must match")
    if top_k < 1 or len(articles) <= top_k:
        raise ValueError("The corpus must contain at least top_k + 1 articles")

    # Find the indices of all articles in the target category.
    target_ids = [i for i, article in enumerate(articles)
                  if article["article_section"] == section]
    if not target_ids:
        raise ValueError(f"No articles found for section: {section}")

    # Iterate over each target article and count how many of its top_k
    # most similar articles belong to the same category.
    total_goods = 0
    for article_id in target_ids:
        # Get the similarity scores for the current article.
        sims = similarity_index[vectors[article_id]]
        # Sort articles by similarity score to rank the most similar results.
        ranked = sorted(
            ((i, score) for i, score in enumerate(sims) if i != article_id),
            key=lambda item: -item[1],
        )

        # Count how many of the top_k articles belong to the target category.
        total_goods += sum(
            articles[i]["article_section"] == section
            for i, _ in ranked[:top_k]
        )

    # ratio_quality = total_goods / (number of target articles * top_k).
    return {
        "section": section,
        "corpus_size": len(articles),
        "num_target_articles": len(target_ids),
        "top_k": top_k,
        "exclude_self": True,
        "total_goods": total_goods,
        "ratio_quality": total_goods / (len(target_ids) * top_k),
    }
