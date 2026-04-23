import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# Load data
df = pd.read_csv("real_data.csv")

df["text"] = df["text"].fillna("")


# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["text"])


# LSA
svd = TruncatedSVD(n_components=100, random_state=42)
lsa_matrix = svd.fit_transform(tfidf_matrix)


# Similarity
similarity_matrix = cosine_similarity(lsa_matrix)


# Evaluation (proxy using component)
def evaluate(df, sim_matrix, k_list=[1, 5, 10]):
    results = {k: 0 for k in k_list}
    mrr_total = 0
    count = 0

    groups = df["component"].fillna("").tolist()

    for i in range(len(df)):
        true_group = groups[i]

        if true_group == "":
            continue

        count += 1

        sims = sim_matrix[i]
        ranked_idx = np.argsort(-sims)

        ranked_idx = ranked_idx[ranked_idx != i]

        found_rank = None

        for rank, idx in enumerate(ranked_idx, start=1):
            if groups[idx] == true_group:
                found_rank = rank
                break

        if found_rank:
            mrr_total += 1 / found_rank

            for k in k_list:
                if found_rank <= k:
                    results[k] += 1

    metrics = {}

    for k in k_list:
        metrics[f"Hit@{k}"] = results[k] / count if count else 0

    metrics["MRR"] = mrr_total / count if count else 0

    return metrics


# Run
metrics = evaluate(df, similarity_matrix)

print("Baseline Results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


# Save
results_df = pd.DataFrame([metrics])
results_df.to_csv("baseline_results.csv", index=False)
