#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC_FILE  = ROOT / "video_metrics_for_ML.csv"
DEST_FILE = ROOT / "video_metrics_for_ML__comments_only.csv"
RESULTS_FIGS = ROOT / "results" / "figures"
RESULTS_FIGS.mkdir(parents=True, exist_ok=True)

if not SRC_FILE.exists():
    raise SystemExit(f"Missing input file: {SRC_FILE}")

df_full = pd.read_csv(SRC_FILE, low_memory=False)
print("Full dataset shape:", df_full.shape)

# Filter to videos with >=1 comment
if "all_comments_total" not in df_full.columns:
    raise SystemExit("Expected column 'all_comments_total' not found.")
df_comm = df_full[df_full["all_comments_total"] > 0].copy()
print("Comments-only dataset shape:", df_comm.shape)

# Create Success Labels (Q75) — views / likes / likes-per-view
if "view_count" in df_comm.columns:
    q75_views = df_comm["view_count"].quantile(0.75)
    df_comm["label_success_views_q75"] = (df_comm["view_count"] >= q75_views).astype(int)
    print("Views Q75 threshold:", q75_views)

if "likes" in df_comm.columns:
    q75_likes = df_comm["likes"].quantile(0.75)
    df_comm["label_success_likes_q75"] = (df_comm["likes"] >= q75_likes).astype(int)
    print("Likes Q75 threshold:", q75_likes)

if {"likes","view_count"}.issubset(df_comm.columns):
    df_comm["likes_per_view_ratio"] = np.where(
        df_comm["view_count"] > 0,
        df_comm["likes"] / df_comm["view_count"],
        np.nan
    )
    q75_lpv = df_comm["likes_per_view_ratio"].quantile(0.75)
    df_comm["label_success_lpv_q75"] = (df_comm["likes_per_view_ratio"] >= q75_lpv).astype(int)
    print("Likes-per-view Q75 threshold:", q75_lpv)

# Log-transform owner engagement share & response rate
df_comm["log_owner_engagement_share"] = np.log1p(df_comm.get("owner_engagement_share", 0.0))
df_comm["log_response_rate"] = np.log1p(df_comm.get("response_rate", 0.0))
print("Added log features: log_owner_engagement_share, log_response_rate")

# Save
df_comm.to_csv(DEST_FILE, index=False)
print("Saved comments-only dataset to:", DEST_FILE)

# Summary
n_videos_comm = df_comm["video_id"].nunique() if "video_id" in df_comm.columns else len(df_comm)
med_owner_share = df_comm.loc[df_comm["all_comments_total"]>0, "owner_engagement_share"].median()
med_resp_rate   = df_comm.loc[df_comm["total_top_comments"]>0, "response_rate"].median()

print("\n=== Comments-only Summary ===")
print(f"- unique videos: {n_videos_comm}")
print(f"- median owner_engagement_share: {med_owner_share:.3f}")
print(f"- median response_rate        : {med_resp_rate:.3f}")

# EDA plots (saved)
def _hist(series, title, xlabel, fname):
    plt.figure(figsize=(4,3))
    series.dropna().hist(bins=50)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_FIGS / fname, dpi=150)
    plt.close()

def _scatter(x, y, title, xlabel, ylabel, fname):
    plt.figure(figsize=(4,3))
    plt.scatter(x, y, s=8, alpha=0.5)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(RESULTS_FIGS / fname, dpi=150)
    plt.close()

_hist(df_comm["owner_engagement_share"], "Owner Engagement Share (comments-only)",
      "owner_engagement_share", "hist_owner_share_comments_only.png")
_hist(df_comm["response_rate"], "Response Rate (comments-only)",
      "response_rate", "hist_response_rate_comments_only.png")

if "likes" in df_comm.columns:
    _scatter(df_comm["likes"], df_comm["owner_engagement_share"],
             "Likes vs Owner Engagement (comments-only)", "likes", "owner_engagement_share",
             "scatter_likes_vs_owner_share_comments_only.png")

if "view_count" in df_comm.columns:
    _scatter(df_comm["view_count"], df_comm["owner_engagement_share"],
             "Views vs Owner Engagement (comments-only)", "view_count", "owner_engagement_share",
             "scatter_views_vs_owner_share_comments_only.png")

# Additional log-feature plots
if "likes" in df_comm.columns:
    _scatter(df_comm["likes"], df_comm["log_owner_engagement_share"],
             "Likes vs log Owner Engagement", "likes", "log_owner_engagement_share",
             "scatter_likes_vs_log_owner_share.png")

if "view_count" in df_comm.columns:
    _scatter(df_comm["view_count"], df_comm["log_owner_engagement_share"],
             "Views vs log Owner Engagement", "view_count", "log_owner_engagement_share",
             "scatter_views_vs_log_owner_share.png")

if "likes" in df_comm.columns:
    _scatter(df_comm["likes"], df_comm["log_response_rate"],
             "Likes vs log Response Rate", "likes", "log_response_rate",
             "scatter_likes_vs_log_response_rate.png")

if "view_count" in df_comm.columns:
    _scatter(df_comm["view_count"], df_comm["log_response_rate"],
             "Views vs log Response Rate", "view_count", "log_response_rate",
             "scatter_views_vs_log_response_rate.png")