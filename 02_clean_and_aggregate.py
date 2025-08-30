#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COMMENTS_PATH = ROOT / "scrape_out" / "comments_all.csv"
VIDEOS_PATH   = ROOT / "USvideos_aug2020_public_clean.csv"
OUT_PATH      = ROOT / "video_metrics_for_ML.csv"
RESULTS_FIGS  = ROOT / "results" / "figures"
RESULTS_FIGS.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Load data
# -----------------------------
if not COMMENTS_PATH.exists():
    raise SystemExit(f"Missing {COMMENTS_PATH}")
if not VIDEOS_PATH.exists():
    raise SystemExit(f"Missing {VIDEOS_PATH}")

use_cols = [
    "video_id","comment_id","parent_comment_id","is_top_level",
    "author_channel_id","is_owner_comment","published_at","like_count","text_original"
]
comments = pd.read_csv(COMMENTS_PATH, usecols=use_cols, low_memory=False)
videos = pd.read_csv(VIDEOS_PATH, low_memory=False)

# -----------------------------
# 2) Audience-only dedup (keep all owner replies)
# -----------------------------
comments["published_at"] = pd.to_datetime(comments["published_at"], errors="coerce", utc=True)

audience = (
    comments[comments["is_owner_comment"] == 0]
    .sort_values(["video_id","published_at","comment_id"])
    .drop_duplicates(subset=["video_id","text_original"], keep="first")
)
owner = comments[comments["is_owner_comment"] == 1]
comments_dedup = pd.concat([audience, owner], ignore_index=True).reset_index(drop=True)

print("Rows before:", len(comments))
print("Rows after audience-only dedup:", len(comments_dedup))

# -----------------------------
# 3) Aggregate per-video stats
# -----------------------------
comments_dedup["is_top_level"] = comments_dedup["is_top_level"].astype(int)
top = comments_dedup[comments_dedup["is_top_level"] == 1].copy()
rep = comments_dedup[comments_dedup["is_top_level"] == 0].copy()

# Replies
rep_stats = (
    rep.groupby("video_id")["is_owner_comment"]
       .agg(owner_replies="sum", total_replies="count")
       .reset_index()
)

# Top-level
top_stats = (
    top.assign(is_owner_top=top["is_owner_comment"].astype(int))
       .groupby("video_id")["is_owner_top"]
       .agg(owner_top_comments="sum", total_top_comments="count")
       .reset_index()
)

# Response rate: threads with ≥1 owner reply / total threads
if not rep.empty:
    owner_reply_by_parent = (
        rep.groupby(["video_id","parent_comment_id"])["is_owner_comment"]
           .max()
           .reset_index()
           .rename(columns={"is_owner_comment":"owner_replied_under_parent"})
    )
    tl_with_owner_reply = (
        top[["video_id","comment_id"]]
        .merge(
            owner_reply_by_parent,
            left_on=["video_id","comment_id"],
            right_on=["video_id","parent_comment_id"],
            how="left"
        )
        .assign(owner_replied_under_parent=lambda df: df["owner_replied_under_parent"].fillna(0).astype(int))
        .groupby("video_id")["owner_replied_under_parent"]
        .agg(threads_with_owner_reply="sum")
        .reset_index()
    )
else:
    tl_with_owner_reply = top.groupby("video_id").size().rename("threads_with_owner_reply").reset_index()
    tl_with_owner_reply["threads_with_owner_reply"] = 0

# Combine engagement stats
eng = (
    top_stats
    .merge(rep_stats, on="video_id", how="outer")
    .merge(tl_with_owner_reply, on="video_id", how="left")
    .fillna(0)
)

eng["owner_comments_total"]    = eng["owner_top_comments"] + eng["owner_replies"]
eng["audience_comments_total"] = (
    (eng["total_top_comments"] - eng["owner_top_comments"]) +
    (eng["total_replies"] - eng["owner_replies"])
)
eng["all_comments_total"]      = eng["owner_comments_total"] + eng["audience_comments_total"]

eng["owner_engagement_share"] = np.where(
    eng["all_comments_total"] > 0,
    eng["owner_comments_total"] / eng["all_comments_total"],
    0.0
)

eng["response_rate"] = np.where(
    eng["total_top_comments"] > 0,
    eng["threads_with_owner_reply"] / eng["total_top_comments"],
    0.0
)

# -----------------------------
# 4) Join with video metadata + coverage
# -----------------------------
keep = [
    "video_id","channelId","title","categoryId","publishedAt",
    "view_count","likes","comment_count","category_name","interactive_likely"
]
keep = [k for k in keep if k in videos.columns]
v = videos[keep].copy()

if "comment_count" in v.columns:
    coverage = eng[["video_id","all_comments_total"]].rename(columns={"all_comments_total":"sampled_comments"})
    v = v.merge(coverage, on="video_id", how="left").fillna({"sampled_comments":0})
    v["sample_coverage_ratio"] = np.where(
        v["comment_count"].astype(float) > 0,
        v["sampled_comments"] / v["comment_count"].astype(float),
        np.nan
    )
else:
    v["sampled_comments"] = np.nan
    v["sample_coverage_ratio"] = np.nan

ml = v.merge(eng, on="video_id", how="left").fillna(0)

# -----------------------------
# 5) Save + snapshot
# -----------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
ml.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}  (rows: {len(ml)}, cols: {ml.shape[1]})")

done_videos = ml["video_id"].nunique()
print("\nSnapshot:")
print(f"- unique videos with any comments scraped: {done_videos}")
if done_videos:
    try:
        med_share = ml.loc[ml["all_comments_total"]>0, "owner_engagement_share"].median()
    except Exception:
        med_share = float("nan")
    try:
        med_resp = ml.loc[ml["total_top_comments"]>0, "response_rate"].median()
    except Exception:
        med_resp = float("nan")
    covered = ml.get("sample_coverage_ratio")
    if covered is not None:
        covered = covered.replace([np.inf,-np.inf], np.nan).dropna()
        med_cov = covered.median() if len(covered) else float("nan")
    else:
        med_cov = float("nan")
    print(f"- median owner_engagement_share: {med_share:.3f}" if pd.notnull(med_share) else "- median owner_engagement_share: n/a")
    print(f"- median response_rate         : {med_resp:.3f}" if pd.notnull(med_resp) else "- median response_rate: n/a")
    print(f"- median sample coverage ratio : {med_cov:.3f}" if pd.notnull(med_cov) else "- median sample coverage ratio: n/a")

# -----------------------------
# 6) EDA helper plots (optional)
# -----------------------------
def _safe_hist(series, title, xlabel, fname):
    plt.figure(figsize=(6,4))
    series.dropna().hist(bins=50)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_FIGS / fname, dpi=150)
    plt.close()

def _safe_scatter(x, y, title, xlabel, ylabel, fname):
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, s=8)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(RESULTS_FIGS / fname, dpi=150)
    plt.close()

print("\nPlotting EDA... (saved to results/figures)")
# owner_engagement_share
_safe_hist(ml.loc[ml["all_comments_total"]>0, "owner_engagement_share"],
           "Distribution of Owner Engagement Share", "owner_engagement_share",
           "hist_owner_engagement_share.png")

# response_rate
_safe_hist(ml.loc[ml["total_top_comments"]>0, "response_rate"],
           "Distribution of Response Rate", "response_rate",
           "hist_response_rate.png")

# all_comments_total
_safe_hist(ml["all_comments_total"],
           "Distribution of All Comments (after audience de-dup)",
           "all_comments_total",
           "hist_all_comments_total.png")

# likes vs owner_engagement_share
if "likes" in ml.columns:
    _safe_scatter(ml["likes"], ml["owner_engagement_share"],
                  "Likes vs Owner Engagement Share", "likes", "owner_engagement_share",
                  "scatter_likes_vs_owner_share.png")

# views vs owner_engagement_share
if "view_count" in ml.columns:
    _safe_scatter(ml["view_count"], ml["owner_engagement_share"],
                  "Views vs Owner Engagement Share", "view_count", "owner_engagement_share",
                  "scatter_views_vs_owner_share.png")

# by interactive flag (simple whisker-like with medians)
if "interactive_likely" in ml.columns:
    flags = [False, True]
    xs = []; medians = []; q1s = []; q3s = []
    for flag in flags:
        vals = ml.loc[ml["interactive_likely"]==flag, "owner_engagement_share"].dropna().values
        if len(vals):
            q1, q2, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        else:
            q1=q2=q3=np.nan
        xs.append(flag); medians.append(q2); q1s.append(q1); q3s.append(q3)

    plt.figure(figsize=(6,4))
    for i, flag in enumerate(flags):
        vals = ml.loc[ml["interactive_likely"]==flag, "owner_engagement_share"].dropna().values
        if len(vals):
            plt.scatter([i]*len(vals), vals, s=5, alpha=0.08)
        if not np.isnan(medians[i]):
            plt.plot([i-0.2, i+0.2], [medians[i], medians[i]])
    plt.xticks([0,1], ["Non-interactive","Interactive"])
    plt.title("Owner Engagement Share by Interactive-Likely Category")
    plt.ylabel("owner_engagement_share")
    plt.tight_layout()
    plt.savefig(RESULTS_FIGS / "owner_share_by_interactive.png", dpi=150)
    plt.close()