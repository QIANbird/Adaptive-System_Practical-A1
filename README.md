# Practical Assignment 1 — News Similarity

News similarity experiments adapted from the course Gensim and NLTK examples. Person A owns the shared pipeline and Tasks 1–2. Person B can reuse the same experiment scripts for Sports in Task 3.

Current status: the basic pipeline is implemented. Full-dataset experiments, result analysis, and report writing are still pending. Assignment materials and dependencies are stored in docs/; the dataset is stored in data/.

## Project structure

```text
Practical Assignment_1/
├── docs/
│   ├── A1_requirement.txt       # Original assignment requirements
│   ├── Adaptive System_PA1.pdf  # Work allocation
│   └── requirements.txt         # Pinned Python dependencies
├── data/
│   └── news2.csv                # Original dataset and default input
├── preprocessing.py             # CSV loading, preprocessing, Dictionary, and BoW
├── evaluation.py                # Shared top-10 evaluation
├── tfidf_experiment.py          # Task 1; also supports Task 3
├── lda_experiment.py            # Task 2; also supports Task 3
├── results/                     # Experiment metrics and case analysis
├── report/                      # Report and prototype
└── README.md
```

## Environment setup

Install Git and Python 3.12 on your computer before starting. Cloning the repository downloads the project files, but does not install Python, dependencies, or NLTK data. Each team member creates their own local environment once.

Clone the repository and enter the cloned directory. Run all commands below from the directory containing this README.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r docs/requirements.txt
python -m nltk.downloader stopwords
```

### Windows (Git Bash)

With Python 3.12 and the Python launcher installed:

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r docs/requirements.txt
python -m nltk.downloader stopwords
```

If `py` is unavailable, use `python -m venv .venv` after checking that `python --version` reports Python 3.12.

### Existing local environment (Person A only)

On the original development machine, the environment in the parent directory can still be reused:

```bash
source ../.venv/bin/activate
```

This parent environment is outside the repository and will not exist in a teammate's fresh clone.

After setup, activate the appropriate environment in each new terminal session before running experiments. Repeat dependency installation after pulling changes to `docs/requirements.txt`.

Download stopwords once during setup. Experiment scripts do not download resources automatically or change SSL certificate verification. The dependency file pins the model libraries and numerical dependencies to the versions verified in the existing environment.

## Running experiments

Start with a small smoke check. These runs are for checking the pipeline and must not be used as assignment results:

```bash
python tfidf_experiment.py --limit 100
python lda_experiment.py --limit 100
```

Run Tasks 1–2 on the full dataset and save the JSON output:

```bash
python tfidf_experiment.py > results/task1_tfidf_food.json
python lda_experiment.py > results/task2_lda_food.json
```

Person B can use the same scripts for Task 3:

```bash
python tfidf_experiment.py --section Sports > results/task3_tfidf_sports.json
python lda_experiment.py --section Sports > results/task3_lda_sports.json
```

The default dataset is `data/news2.csv`, resolved relative to the script directory. To use a different file, pass `--data /path/to/news2.csv`. The `--limit` option reads only the first N raw rows before cleaning; the cleaned sample may contain fewer than N articles and must still contain at least 11 articles and include the selected category.

Each successful run prints the model name, category, corpus size, number of target articles, top-k setting, self-exclusion policy, total relevant matches, quality ratio, and two execution times in seconds. LDA output also records its required model parameters. A non-null `sample_limit` identifies a sample run.

## Shared experiment conventions

- Build text vectors from `description` only. Use `article_section` only to select query articles and evaluate results. Do not include headlines or tags in Tasks 1–3 models.
- Follow the course preprocessing sequence: `lower().split()` → NLTK English stopwords → PorterStemmer → Dictionary → BoW. No additional punctuation cleanup or rare-word filtering is applied.
- Use identical preprocessing for TF-IDF and LDA. Train LDA on BoW with `num_topics=30, passes=2, random_state=30`; keep other model settings at their library defaults.
- The actual dataset label is `Food & Drink`, corresponding to “Food and Drink” in the assignment.
- Search the entire corpus, exclude the query article itself, and take the top 10 remaining articles. Break score ties by CSV row order. Self-exclusion is a project convention because the assignment does not specify this detail.
- Clean articles in `load_articles` before tokenization or vector construction: remove missing, empty, or whitespace-only descriptions, then deduplicate by the exact description after trimming leading and trailing whitespace. Keep the first occurrence in CSV order, even if later occurrences have different headlines, tags, or categories. Case, punctuation, and internal whitespace differences remain distinct. The original CSV is not modified.
- Both models and evaluation use the same cleaned article list. Corpus size, target article count, and the quality denominator therefore refer to the cleaned data. Tokenization preserves this list's order and length; descriptions containing only stopwords can still produce empty token lists. Record the cleaning policy and removed counts in the final report.
- Calculate `ratio_quality = total_goods / (num_target_articles * 10)`, equivalent to mean Precision@10 for queries in the selected category.
- Run each experiment as a separate process. Model creation time starts near the top of the entry script and includes subsequent imports, argument parsing, data loading, preprocessing, training, vector transformation, and MatrixSimilarity index construction. It excludes Python interpreter startup. Evaluation time covers the shared evaluation function separately. JSON printing is outside both measured stages.
- For formal comparisons, use the full dataset without `--limit`, run in the same environment, and record execution conditions.

## Connection to the course examples

- `preprocessing.py` adapts `vectormodel_stem_example.py`, replacing the example document list with CSV descriptions.
- `tfidf_experiment.py` retains TfidfModel and MatrixSimilarity from `tfidf_example.py`.
- `lda_experiment.py` retains LdaModel and MatrixSimilarity from `lda_example.py`, with assignment-specific parameters.
- `evaluation.py` extends the examples' single-document query and sorting into the assignment's query loop and category counting.
- `wordnet_examples.py` is not needed for Tasks 1–2.

## Next steps

- Run and verify full-dataset metrics and both execution times for Tasks 1–2.
- Inspect successful and unsuccessful retrieval examples and explain differences between TF-IDF and LDA.
- Reproduce results with Person B and keep Task 3 data and evaluation conventions consistent.
- Complete the report, Task 4 tagging proposal, and Task 5 prototype where applicable.

## What to commit

Commit the source files, README, `docs/requirements.txt`, `data/news2.csv`, and the directory placeholders. The shared dataset is required to run the default commands. Commit report files and selected experiment results as they become available.

Do not commit a local virtual environment, Python caches, or IDE-specific settings. `.gitignore` excludes `.venv/`, `__pycache__/`, and common local editor settings. Python packages are installed from the dependency file on each computer; the virtual environment itself is not portable.

The existing environment and smoke checks were verified on macOS with Python 3.12.13. Windows setup instructions are provided, but execution on Windows has not yet been verified.
