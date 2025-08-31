import os
import sqlite3
import yaml
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

# ---------- load configuration ----------
def load_config(path: str = "config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # merge external weights file if specified
    if cfg.get("weights_yaml") and os.path.isfile(cfg["weights_yaml"]):
        with open(cfg["weights_yaml"], encoding="utf-8") as f2:
            cfg["weights"].update(yaml.safe_load(f2))
    return cfg

CFG = load_config()

# ---------- extract parameters ----------
INPUT_FILE   = CFG["input_csv"]
CACHE_DB     = CFG["cache_sqlite"]
MODEL_NAME   = CFG["model_name"]
MAX_LEN      = CFG["max_length"]
BATCH_SIZE   = CFG["batch_size"]
DEVICE       = torch.device(CFG["device"])
WEIGHTS      = CFG["weights"]
SCORE_MIN    = CFG["score_range"]["min"]
SCORE_MAX    = CFG["score_range"]["max"]
OUTPUT_SFX   = CFG["output_suffix"]

# ---------- helpers ----------
ID2LABEL  = {0: "Negative", 1: "Neutral", 2: "Positive"}
SCORE_MAP = {"Negative": -1, "Neutral": 0, "Positive": 1}

def init_cache():
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                comment_text TEXT PRIMARY KEY,
                label TEXT,
                score INTEGER
            )
        """)

def lookup(text: str):
    with sqlite3.connect(CACHE_DB) as conn:
        cur = conn.execute("SELECT label, score FROM cache WHERE comment_text=?", (text,))
        return cur.fetchone()

def save(text: str, label: str, score: int):
    with sqlite3.connect(CACHE_DB) as conn:
        conn.execute("INSERT OR IGNORE INTO cache VALUES (?,?,?)", (text, label, score))

# ---------- main pipeline ----------
def main():
    init_cache()

    # 1. load CSV
    use_cols = ["VideoID", "VideoTitle", "AuthorChannelID", "AuthorName",
                "CommentText", "Likes", "Replies"]
    df = pd.read_csv(INPUT_FILE, usecols=use_cols, encoding="macroman")
    df["AuthorChannelID"] = df["AuthorChannelID"].astype("category")
    df["AuthorName"]      = df["AuthorName"].astype("category")

    # 2. identify comments to be processed
    todo_mask  = df["CommentText"].apply(lambda t: lookup(str(t).strip()) is None)
    todo_texts = df.loc[todo_mask, "CommentText"].astype(str).str.strip().tolist()
    print(f"Total rows: {len(df)} | Rows to infer: {len(todo_texts)}")

    if todo_texts:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            truncation_side="left",
            pad_token="</s>"
        )
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()

        ds = Dataset.from_dict({"text": todo_texts})

        def tokenize(batch):
            return tokenizer(batch["text"],
                             padding=True,
                             truncation=True,
                             max_length=MAX_LEN)
        ds = ds.map(tokenize, batched=True, num_proc=1)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        loader = torch.utils.data.DataLoader(ds,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=0)

        offset = 0
        for batch in tqdm(loader, desc="Sentiment"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
            preds = logits.argmax(-1).cpu().tolist()

            bs    = len(preds)
            texts = todo_texts[offset: offset + bs]
            offset += bs
            for text, p in zip(texts, preds):
                label = ID2LABEL[p]
                score = SCORE_MAP[label]
                save(text, label, score)
    else:
        print("All comments found in cache; skipping inference.")

    # 3. merge cache results
    def get_result(text):
        res = lookup(str(text).strip())
        return res if res else ("Neutral", 0)

    tqdm.pandas(desc="Cache")
    df[["roberta_label", "roberta_score"]] = (
        df["CommentText"]
        .progress_apply(lambda t: pd.Series(get_result(t)))
    )

    # 4. compute weights & aggregate by author
    author_groups = df.groupby("AuthorChannelID")
    total_ln_sums = {
        author_id: np.log(1 + g["Likes"] + g["Replies"]).sum()
        for author_id, g in author_groups
    }

    df["CommentWeight"] = df.apply(
        lambda row: np.log(1 + row["Likes"] + row["Replies"]) /
                    total_ln_sums[row["AuthorChannelID"]],
        axis=1
    )

    author_stats = []
    for author_id, g in author_groups:
        sentiment_avg   = g["roberta_score"].mean()
        pos_ratio       = (g["roberta_score"] > 0).mean()
        interaction_sum = (g["Likes"] + g["Replies"]).sum()

        user_counts = g["AuthorName"].value_counts()
        repeat      = (user_counts >= 2).sum()
        total       = len(user_counts)
        retention   = repeat / total if total else 0

        author_stats.append({
            "AuthorChannelID": author_id,
            "AuthorName": g["AuthorName"].iloc[0],
            "TotalComments": len(g),
            "UniqueUsers": total,
            "RepeatUsers": repeat,
            "RetentionRate": retention,
            "SentimentAvg": sentiment_avg,      # [-1,1]
            "PosRatio": pos_ratio,
            "InteractionSum": interaction_sum,
        })

    rank_df = pd.DataFrame(author_stats)
    metric_cols = ["SentimentAvg", "PosRatio", "InteractionSum", "RetentionRate"]
    rank_df[metric_cols] = MinMaxScaler().fit_transform(rank_df[metric_cols])

    rank_df["CompositeScore"] = (
        rank_df["SentimentAvg"]  * WEIGHTS["SentimentAvg"] +
        rank_df["PosRatio"]      * WEIGHTS["PosRatio"] +
        rank_df["InteractionSum"]* WEIGHTS["InteractionSum"] +
        rank_df["RetentionRate"] * WEIGHTS["RetentionRate"]
    ).round(4)

    # 5. final 1-100 scaling
    rank_df = rank_df.sort_values("CompositeScore", ascending=False).reset_index(drop=True)
    rank_df["SentimentScore_1_100"] = (
        (rank_df["SentimentAvg"] + 1) / 2 * (SCORE_MAX - SCORE_MIN) + SCORE_MIN
    ).round(2)

    print("\n🎉 Top 10 Creators (SentimentScore 1-100)")
    print(rank_df[["AuthorName", "SentimentScore_1_100", "CompositeScore"]].head(10))

    # 6. save results
    out_file = os.path.splitext(INPUT_FILE)[0] + OUTPUT_SFX
    rank_df.to_excel(out_file, index=False,
                     columns=["AuthorChannelID", "AuthorName",
                              "TotalComments", "UniqueUsers", "RepeatUsers",
                              "RetentionRate", "SentimentScore_1_100",
                              "PosRatio", "InteractionSum", "CompositeScore"])
    print(f"\nFull ranking saved to {out_file}")

if __name__ == "__main__":
    main()