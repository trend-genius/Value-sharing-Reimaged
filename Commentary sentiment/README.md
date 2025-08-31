# 📊 Video Comment Sentiment Analysis

**Analyze videos and creators through their comments!**  
Powered by Hugging Face’s multilingual sentiment model, this tool automatically:

- Scores sentiment on GPU  
- Deduplicates via cache—subsequent runs finish in seconds  
- Computes multi-dimensional creator rankings (sentiment, engagement, retention)  
- Exports an Excel leaderboard with SentimentScore on a 1–100 scale

---

## 🚀 Quick Start

1. Prepare your data as `input.csv` with these **required columns**:  
   `VideoID, VideoTitle, AuthorChannelID, AuthorName, CommentText, Likes, Replies`

2. Edit `config.yaml` in the same folder—just change `input_csv:` to your file path; everything else can stay default.

3. Run from the project root:
   ```bash
   python main.py
   ```

On first launch the model (~1 GB) is downloaded automatically.  
When finished you’ll get:

- `<input_basename>_creator_ranking.xlsx` — complete leaderboard with 1–100 sentiment scores  
- `sentiment_cache.sqlite` — local cache for instant re-runs

---

## 📋 Leaderboard Fields

| Column | Meaning | Range |
|---|---|---|
| **SentimentScore_1_100** | Average sentiment (1 = very negative, 100 = very positive) | 1–100 |
| **CompositeScore** | Overall creator influence score | 0–1 |
| **RetentionRate** | Viewer return / repeat-comment rate | 0–1 |
| **InteractionSum** | Total interactions (likes + replies), normalized | 0–1 |

---

## ⚙️ Customization

### Change weights or file paths  
Edit **`config.yaml`**:

```yaml
weights:
  SentimentAvg: 0.4   # increase sentiment weight
  PosRatio: 0.2
  InteractionSum: 0.2
  RetentionRate: 0.2

input_csv: "D:/my_comments.csv"
```

---

## 🚨 Common Errors & Fixes

| Error | Cause | Solution |
|---|---|---|
| `CUDA out of memory` | `batch_size` too large | Lower `batch_size` in `config.yaml` (e.g., 64/32) |
| `RuntimeError: CUDA error: device-side assert` | Tensor type mismatch | Run `CUDA_LAUNCH_BLOCKING=1 python main.py` to see the full stack, then cast tensors to `.long()` as needed |
| `ImportError: cannot import name 'AutoModelForSequenceClassification'` | Outdated transformers | `pip install -U transformers` or check your virtual env |
| `FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'` | Wrong working directory or path | Ensure `config.yaml` sits next to `main.py`, or use an absolute path |

---

## 📄 License

m13052784890@163.com

Feel free to open issues or PRs!