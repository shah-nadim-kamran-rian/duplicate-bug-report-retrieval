import os
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text):
    return str(text).lower()


def load_data(path):
    df = pd.read_csv(path)

    if "text" not in df.columns:
        if "summary" in df.columns and "description" in df.columns:
            df["text"] = df["summary"].fillna("") + " " + df["description"].fillna("")
        else:
            raise ValueError(
                "Input CSV must contain either 'text' or both 'summary' and 'description'."
            )
    else:
        df["text"] = df["text"].fillna("")

    df["text"] = df["text"].apply(preprocess)

    required_meta = ["product", "component"]
    for col in required_meta:
        if col not in df.columns:
            raise ValueError(
                f"Input CSV must contain '{col}' for proxy-group evaluation."
            )

    df["product"] = df["product"].fillna("").astype(str).str.strip().str.lower()
    df["component"] = df["component"].fillna("").astype(str).str.strip().str.lower()

    return df


def compute_tfidf(df):
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(df["text"])
    return cosine_similarity(X)


def compute_lsa(df):
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(df["text"])
    svd = TruncatedSVD(n_components=16, random_state=42)
    X_reduced = svd.fit_transform(X)
    return cosine_similarity(X_reduced)


def compute_metadata_similarity(df):
    component_match = (
        df["component"].values[:, None] == df["component"].values[None, :]
    ).astype(float)

    np.fill_diagonal(component_match, 0.0)
    return component_match


def normalize_matrix(sim_matrix):
    sim_matrix = sim_matrix.copy()
    min_val = sim_matrix.min()
    max_val = sim_matrix.max()

    if max_val - min_val < 1e-12:
        return np.zeros_like(sim_matrix)

    return (sim_matrix - min_val) / (max_val - min_val)


def compute_sentence_embeddings(df, model_name):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=16,
        show_progress_bar=True
    )
    return cosine_similarity(embeddings)


def build_proxy_groups(df):
    return df["product"].tolist()


def evaluate_ranking(sim_matrix, groups, ks=(1, 5, 10)):
    n = sim_matrix.shape[0]

    mrr_list = []
    hit_at = {k: [] for k in ks}
    recall_at = {k: [] for k in ks}

    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1

        ranked_idx = np.argsort(-sims)
        true_group = groups[i]

        relevant = [j for j in range(n) if j != i and groups[j] == true_group]
        n_relevant = len(relevant)

        if n_relevant == 0:
            continue

        rr = 0.0
        for rank, j in enumerate(ranked_idx, start=1):
            if groups[j] == true_group:
                rr = 1.0 / rank
                break
        mrr_list.append(rr)

        for k in ks:
            top_k = ranked_idx[:k]
            hits = sum(1 for j in top_k if groups[j] == true_group)

            hit_at[k].append(1.0 if hits > 0 else 0.0)
            recall_at[k].append(hits / n_relevant)

    results = {
        "mrr": float(np.mean(mrr_list)) if mrr_list else 0.0
    }

    for k in ks:
        results[f"hit@{k}"] = float(np.mean(hit_at[k])) if hit_at[k] else 0.0
        results[f"recall@{k}"] = float(np.mean(recall_at[k])) if recall_at[k] else 0.0

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--semantic", default="none")
    parser.add_argument(
        "--semantic-model",
        default="sentence-transformers/all-mpnet-base-v2"
    )
    args = parser.parse_args()

    print("Loading data...")
    df = load_data(args.csv)

    
    df = df.head(10000).copy()

    groups = build_proxy_groups(df)

    print("Running TF-IDF...")
    tfidf_sim = compute_tfidf(df)
    tfidf_results = evaluate_ranking(tfidf_sim, groups)

    print("Running LSA...")
    lsa_sim = compute_lsa(df)
    lsa_results = evaluate_ranking(lsa_sim, groups)

    print("Running Metadata Similarity...")
    meta_sim = compute_metadata_similarity(df)
    meta_results = evaluate_ranking(meta_sim, groups)

    print("Running Hybrid Model...")
    tfidf_norm = normalize_matrix(tfidf_sim)
    meta_norm = normalize_matrix(meta_sim)

    
    alpha = 0.85
    gamma = 0.15

    hybrid_sim = alpha * tfidf_norm + gamma * meta_norm
    hybrid_results = evaluate_ranking(hybrid_sim, groups)

    results = {
        "tfidf_mrr": tfidf_results["mrr"],
        "tfidf_hit@1": tfidf_results["hit@1"],
        "tfidf_recall@5": tfidf_results["recall@5"],
        "tfidf_recall@10": tfidf_results["recall@10"],

        "lsa_mrr": lsa_results["mrr"],
        "lsa_hit@1": lsa_results["hit@1"],
        "lsa_recall@5": lsa_results["recall@5"],
        "lsa_recall@10": lsa_results["recall@10"],

        "meta_mrr": meta_results["mrr"],
        "meta_hit@1": meta_results["hit@1"],
        "meta_recall@5": meta_results["recall@5"],
        "meta_recall@10": meta_results["recall@10"],

        "hybrid_mrr": hybrid_results["mrr"],
        "hybrid_hit@1": hybrid_results["hit@1"],
        "hybrid_recall@5": hybrid_results["recall@5"],
        "hybrid_recall@10": hybrid_results["recall@10"],
    }

    if args.semantic == "sentence":
        print("Running Sentence Transformer...")
        sem_sim = compute_sentence_embeddings(df, args.semantic_model)
        sem_results = evaluate_ranking(sem_sim, groups)

        results["mpnet_mrr"] = sem_results["mrr"]
        results["mpnet_hit@1"] = sem_results["hit@1"]
        results["mpnet_recall@5"] = sem_results["recall@5"]
        results["mpnet_recall@10"] = sem_results["recall@10"]

    os.makedirs(args.outdir, exist_ok=True)
    out = pd.DataFrame([results])
    out.to_csv(f"{args.outdir}/main_results.csv", index=False)

    print("Done. Results saved.")


if __name__ == "__main__":
    main()
