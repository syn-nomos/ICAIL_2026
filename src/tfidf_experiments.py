import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse import vstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

# =========================
# CONFIG
# =========================
EXPERIMENTS = {
    "analyzers": ["word", "char"],
    "ngram_ranges": [(1,1), (1,2), (1,3), (2,2), (4,4), (5,5)],
    "min_dfs": [1, 2, 3, 4],
    "max_dfs": [0.95, 0.98, 0.99, 1.0],
}

ALLOWED_NGRAMS = {
     #"char": [(4, 4), (5, 5)],
     #"word": [(1, 1), (1, 2), (1, 3), (2, 2)],
}

TOP_K_VALUES = [1, 3, 5, 10, 15, 20, 50, 100]
TOP_K_MAX = 1000 #adjust this for quick run

# =========================
# PATHS
# =========================
# --- dataset split dirs (multiple JSON docs per split) ---
DATASET_DIRS = {
    "train": "./EURLEX57K/dataset/train/",
    "dev":   "./EURLEX57K/dataset/dev/",
    "test":  "./EURLEX57K/dataset/test/",
}


# Each line should contain at least: {"label": "<concept_id>", "label": "...", "LLM_Response_text": "..."}
LLM_LOOKUP_JSONL = "./eurlex/label_description_file.jsonl"
USE_TFIDF_STRENGTHENING = True

# output
OUTPUT_DIR = "./eurlex/metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# UTILS
# =========================
def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[.,;:!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def doc_text(item: dict) -> str:
    parts = []
    for key in ["title", "main_body"]: #, "recitals", "annexes", "preamble"
        if key in item and item[key]:
            if isinstance(item[key], list):
                parts.extend(x for x in item[key] if isinstance(x, str))
            elif isinstance(item[key], str):
                parts.append(item[key])
    return " ".join(clean_text(p) for p in parts)

def build_vectorizer(analyzer, ngram_range, min_df, max_df):
    stop_words = 'english' if analyzer == 'word' else None
    kwargs = dict(
        dtype=np.float32,
        analyzer=analyzer,
        ngram_range=ngram_range,
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df,
        max_features=None
    )
    if USE_TFIDF_STRENGTHENING:
        kwargs.update(dict(
            sublinear_tf=True,
            smooth_idf=True,
            use_idf=True,
            norm='l2'
        ))
    return TfidfVectorizer(**kwargs)

def apply_idf_power(X_sparse, vec, alpha: float = 1.0):
    if (not USE_TFIDF_STRENGTHENING) or (alpha == 1.0):
        return X_sparse
    idf = vec._tfidf.idf_
    scale = np.power(idf, alpha - 1.0).astype(np.float32)
    return X_sparse @ sp.diags(scale, 0, format='csr')

def dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

LABEL_STOPWORDS = {

    "legal","encompasses","refers","concept","regulations","various",
    "rights","including","compliance","laws","issues","law",
    "additionally","related","may", "can", "often", "typically", "generally",
    "various", "certain", "specific", "different","within",
    "system", "process", "mechanism", "framework", "approach", "context",
    "area", "field", "scope", "domain", "structure", "refers", "covers",
    "concerns", "addresses", "relates","is","are","be","being"

    # "refers", "covers", "concerns", "addresses", "relates",
    # "aims", "focuses", "defines", "establishes", "regulates",
    # "is","are","be","being","includes","involves","may", "can", "often", "typically", "generally",
    # "various", "certain", "specific", "different"
}
def remove_label_stopwords(text: str) -> str:
    toks = text.split()
    toks = [t for t in toks if t not in LABEL_STOPWORDS]
    return " ".join(toks)

# =========================
# 1) LOAD LOOKUP JSONL
# =========================
def load_llm_lookup(jsonl_path: str, use_label_name_prefix: bool = True):
    """
    Lookup file lines (as in your attached file):
      {"concept_id":"100142","label":"04 Politics","LLM_Response_text":"..."}
    Returns:
      id_to_desc: {concept_id: cleaned_text}
    """
    id_to_desc = {}
    missing_desc = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            #use concept_id as the key
            cid = obj.get("concept_id")
            if cid is None:
                continue
            cid = str(cid).strip()

            desc = obj.get("LLM_Response_text")
            desc = remove_label_stopwords(desc)
            if not desc:
                missing_desc += 1
                continue

            
            label_name = obj.get("label")

            if use_label_name_prefix and label_name:
                final = clean_text(f"{label_name} {desc}")
            else:
                final = clean_text(desc)

            if final and "[error]" not in final.lower() and cid not in id_to_desc:
                id_to_desc[cid] = final

    return id_to_desc, {"unique_ids": len(id_to_desc), "missing_desc_lines": missing_desc}

# =========================
# 2) READ DATASET DOCS BY SPLIT
# =========================
def read_dataset_split(split_dir: str):
    """
    Reads dataset documents from a split directory.
    Expects each doc JSON to have at least:
      - "concepts": list of concept IDs (strings)
    Optionally:
      - "celex_id"
      - text fields used by doc_text()
    """
    docs_text = []
    docs_labels = []
    celex_ids = []

    for fn in tqdm(os.listdir(split_dir), desc=f"Reading split: {Path(split_dir).name}"):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(split_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)

        celex_ids.append(item.get("celex_id", fn))
        docs_text.append(clean_text(doc_text(item)))
        docs_labels.append(item.get("concepts", []))

    return celex_ids, docs_text, docs_labels

# =========================
# 3) METRICS
# =========================
def compute_propensities_from_freq(label_freq: dict, A=0.55, B=1.5, C=1.0):
    prop = {}
    for l, n in label_freq.items():
        prop[l] = 1.0 / (1.0 + C * (float(n) + B) ** (-A))
    return prop

def psp_at_k_for_instance(gold_labels, ranked_labels, propensities, k):
    top = dedup_keep_order(ranked_labels[:k])
    if not top:
        return 0.0
    gold = set(gold_labels)
    s = 0.0
    for l in top:
        if l in gold:
            p = propensities.get(l, 0.0)
            if p > 0:
                s += 1.0 / p
    return s / float(len(top))

def psp_at_k_dataset(gold, preds, propensities, k):
    vals = [psp_at_k_for_instance(y, yhat, propensities, k) for y, yhat in zip(gold, preds)]
    return float(np.mean(vals)) if vals else 0.0

def ndcg_at_k(y_true, y_pred, k):
    y_true = set(y_true)
    top = dedup_keep_order(y_pred[:k])
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(top) if p in y_true)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(y_true))))
    return dcg / ideal if ideal > 0 else 0.0

def recall_at_k(y_true, y_pred, k):
    if not y_true:
        return 0.0
    top = set(dedup_keep_order(y_pred[:k]))
    return sum(1 for l in y_true if l in top) / len(y_true)

def precision_at_k(y_true, y_pred, k):
    top = dedup_keep_order(y_pred[:k])
    if not top:
        return 0.0
    y_true = set(y_true)
    hits = sum(1 for l in top if l in y_true)
    return hits / len(top)

# =========================
# 4) MAIN EXPERIMENT
# =========================
def run():
    # Load lookup once
    id_to_desc, stats = load_llm_lookup(LLM_LOOKUP_JSONL, use_label_name_prefix=True)
    print(f"Lookup loaded: {stats['unique_ids']} ids (missing_desc_lines={stats['missing_desc_lines']})")

    # Read dataset splits
    train_celex, train_docs, train_gold = read_dataset_split(DATASET_DIRS["train"])
    test_celex, test_docs, test_gold = read_dataset_split(DATASET_DIRS["test"])

    # label frequencies from TRAIN split
    train_label_freq = Counter()
    for labels in train_gold:
        for cid in labels:
            if isinstance(cid, str):
                train_label_freq[cid] += 1

    prop = compute_propensities_from_freq(train_label_freq)

    # Build label bank = all labels we want to retrieve over
    all_label_ids = dedup_keep_order(list(train_label_freq.keys()) + dedup_keep_order([l for ys in test_gold for l in ys]))

    # Keep only labels that have lookup descriptions
    bank_ids = [cid for cid in all_label_ids if cid in id_to_desc]
    bank_texts = [id_to_desc[cid] for cid in bank_ids]

    missing = len(all_label_ids) - len(bank_ids)
    print(f"Label bank: {len(bank_ids)} labels with descriptions (missing_in_lookup={missing})")

    # TF-IDF grid
    for analyzer in EXPERIMENTS["analyzers"]:
        for ngram_range in EXPERIMENTS["ngram_ranges"]:
            if analyzer in ALLOWED_NGRAMS and ngram_range not in ALLOWED_NGRAMS[analyzer]:
                continue

            for min_df in EXPERIMENTS["min_dfs"]:
                for max_df in EXPERIMENTS["max_dfs"]:
                    print(f"\n===== analyzer={analyzer} ngram={ngram_range} min_df={min_df} max_df={max_df} =====")

                    vec = build_vectorizer(analyzer, ngram_range, min_df, max_df)
                    vec.fit(train_docs + bank_texts)  # you can also add bank_texts if you want: train_docs + bank_texts

                    label_vecs = vec.transform(bank_texts)
                    doc_vecs = vec.transform(test_docs)

                    label_vecs = apply_idf_power(label_vecs, vec, 1.6)
                    doc_vecs = apply_idf_power(doc_vecs, vec, 1.0)

                    if label_vecs.shape[0] == 0:
                        print("No labels available to score.")
                        continue

                    scores = linear_kernel(doc_vecs, label_vecs)
                    K = min(TOP_K_MAX, scores.shape[1])  # important bound

                    topk_idx = np.argpartition(scores, -K, axis=1)[:, -K:]
                    topk_scores = np.take_along_axis(scores, topk_idx, axis=1)
                    order_within = np.argsort(-topk_scores, axis=1)
                    topk_sorted_idx = np.take_along_axis(topk_idx, order_within, axis=1)
                    topk_sorted_scores = np.take_along_axis(scores, topk_sorted_idx, axis=1)

                    predictions = []
                    for i in range(scores.shape[0]):
                        pred_ids = [bank_ids[j] for j in topk_sorted_idx[i].tolist()]
                        pred_ids = dedup_keep_order(pred_ids)
                        predictions.append({
                            "celex_id": test_celex[i],
                            "gold_labels": test_gold[i],
                            "ranked_labels": pred_ids,
                            "ranked_scores": topk_sorted_scores[i].tolist(),
                        })

                    gold_list = [p["gold_labels"] for p in predictions]
                    preds_list = [p["ranked_labels"] for p in predictions]

                    metrics = {}
                    for k in TOP_K_VALUES:
                        nd = np.mean([ndcg_at_k(g, p, k) for g, p in zip(gold_list, preds_list)])
                        rc = np.mean([recall_at_k(g, p, k) for g, p in zip(gold_list, preds_list)])
                        pr = np.mean([precision_at_k(g, p, k) for g, p in zip(gold_list, preds_list)])
                        ps = psp_at_k_dataset(gold_list, preds_list, prop, k)

                        #print(f"Top-{k}: NDCG={nd:.4f} Recall={rc:.4f} Precision={pr:.4f} PSP={ps:.4f}")
                        metrics[f"Top-{k}"] = {"NDCG": float(nd), "Recall": float(rc), "Precision": float(pr), "PSP": float(ps)}

                    out_name = f"lookup_tfidf_{analyzer}_{ngram_range[0]}_{ngram_range[1]}_min{min_df}_max{max_df}.json"
                    out_path = os.path.join(OUTPUT_DIR, out_name)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2, ensure_ascii=False)
                    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    run()
