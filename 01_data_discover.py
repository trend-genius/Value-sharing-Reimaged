#!/usr/bin/env python3
import os, time, requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# ==== CONFIG ====
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "US_youtube_trending_data.csv"
OUT_DIR = ROOT
AUG_ONLY_OUT = OUT_DIR / "USvideos_aug2020_comments_enabled.csv"
PUBLIC_RAW   = OUT_DIR / "USvideos_aug2020_public.csv"          # before dropna
PUBLIC_CLEAN = OUT_DIR / "USvideos_aug2020_public_clean.csv"    # after dropna
PUBLIC_IDS   = OUT_DIR / "USvideos_aug2020_public_ids.csv"
BAD_IDS      = OUT_DIR / "USvideos_aug2020_bad_ids.csv"

# ==== AUTH ====
load_dotenv(ROOT / ".env")
API_KEY = (os.getenv("YT_API_KEY") or "").strip()
if not API_KEY:
    raise SystemExit("Missing YT_API_KEY. Put it in .env like: YT_API_KEY=AIza...")

BASE = "https://www.googleapis.com/youtube/v3"

# ---------- Helpers ----------
def yt_get(endpoint, params, retries=5):
    q = dict(params); q["key"] = API_KEY
    backoff = 1.0
    for _ in range(retries):
        r = requests.get(f"{BASE}/{endpoint}", params=q, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (403, 429, 500, 503):
            time.sleep(backoff); backoff = min(backoff*2, 30); continue
        raise RuntimeError(f"{endpoint} {r.status_code}: {r.text[:300]}")
    raise RuntimeError(f"{endpoint} failed after {retries} retries")

def filter_public(ids):
    """Return (public_ids, bad_ids) using chunk bisect to isolate any 400-causing IDs."""
    public, bad = [], []

    def check_chunk(chunk):
        nonlocal public, bad
        if not chunk: return
        try:
            j = yt_get("videos", {"part": "status", "id": ",".join(chunk)})
            for it in j.get("items", []):
                if it.get("status", {}).get("privacyStatus") == "public":
                    public.append(it["id"])
        except RuntimeError:
            if len(chunk) == 1:
                bad.append(chunk[0]); return
            mid = len(chunk)//2
            check_chunk(chunk[:mid]); check_chunk(chunk[mid:])

    for i in range(0, len(ids), 50):  # API limit
        check_chunk(ids[i:i+50])
        time.sleep(0.1)
    return sorted(set(public)), sorted(set(bad))

# ---------- Pipeline ----------
def main():
    if not SRC.exists():
        raise SystemExit(f"Missing source file: {SRC}")

    # Load and filter to Aug 2020 & comments-enabled
    df = pd.read_csv(SRC, low_memory=False)
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)

    mask_aug = (df["publishedAt"].dt.year == 2020) & (df["publishedAt"].dt.month == 8)
    aug = df.loc[mask_aug].copy()

    if "comments_disabled" in aug.columns:
        aug = aug[aug["comments_disabled"] == False].copy()

    print("Total August 2020 videos:", int(mask_aug.sum()))
    print("After filtering comment-disabled:", len(aug))
    aug.to_csv(AUG_ONLY_OUT, index=False)

    # Unique IDs to check
    ids = aug["video_id"].astype(str).dropna().unique().tolist()
    print("Unique August IDs to check:", len(ids))

    public_ids, bad_ids = filter_public(ids)
    print("Public now:", len(public_ids))
    print("Bad/invalid IDs (skipped):", len(bad_ids))

    pd.DataFrame({"video_id": public_ids}).to_csv(PUBLIC_IDS, index=False)
    pd.DataFrame({"video_id": bad_ids}).to_csv(BAD_IDS, index=False)

    # Keep only rows whose video_id is currently public
    aug_public = aug[aug["video_id"].isin(public_ids)].copy()
    print("Remaining public videos (rows):", len(aug_public))
    aug_public.to_csv(PUBLIC_RAW, index=False)

    # -------- NEW: drop rows with any missing values --------
    before = len(aug_public)
    aug_clean = aug_public.dropna(how="any").copy()
    after = len(aug_clean)
    print(f"Dropped {before - after} rows with missing values; final rows: {after}")

    aug_clean.to_csv(PUBLIC_CLEAN, index=False)
    print(f"Saved cleaned dataset -> {PUBLIC_CLEAN}")

if __name__ == "__main__":
    main()