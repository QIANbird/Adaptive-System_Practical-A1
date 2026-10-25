"""共享数据处理：沿用 vectormodel_stem_example.py 的预处理步骤。"""

import csv
from pathlib import Path

from gensim import corpora
from nltk import PorterStemmer
from nltk.corpus import stopwords


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "news2.csv"


def load_articles(data_path=DEFAULT_DATA_PATH, limit=None):
    """按 CSV 原顺序读取新闻；limit 只用于小样本冒烟检查。"""
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    with Path(data_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"headline", "tags", "article_section", "description"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain these columns: {sorted(required)}")
        articles = []
        for row in reader:
            articles.append({key: (row.get(key) or "").strip() for key in required})
            if limit is not None and len(articles) >= limit:
                break
    if not articles:
        raise ValueError("The dataset contains no articles")
    return articles


def preprocess_descriptions(articles):
    """小写化 → 空白分词 → 去英文停用词 → Porter 词干提取。"""
    porter = PorterStemmer()
    try:
        stoplist = set(stopwords.words("english"))
    except LookupError as error:
        raise RuntimeError(
            "Missing NLTK stopwords. Run: python -m nltk.downloader stopwords"
        ) from error
    # 保留空描述和重复行，保证文章下标与向量下标始终一致。
    return [
        [porter.stem(word) for word in article["description"].lower().split()
         if word not in stoplist]
        for article in articles
    ]


def build_corpus(articles):
    """返回两个模型共用的 Dictionary 和 BoW 向量。"""
    texts = preprocess_descriptions(articles)
    dictionary = corpora.Dictionary(texts)
    if not dictionary:
        raise ValueError("The dataset contains no usable tokens")
    model_bow = [dictionary.doc2bow(text) for text in texts]
    return dictionary, model_bow
