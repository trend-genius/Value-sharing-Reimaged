README.md
========

YouTube-Trending-Comment-Success-Prediction  
Predict whether a trending YouTube video will become a **“success”** (top 25 % by views, likes or likes-per-view) based **only** on early comment-interaction signals.

---

📁 Project structure
--------------------
```
.
├── config
│   └── predict.yaml               # hyper-parameters & run-time configs
├── data
│   └── data.csv                   # raw YouTube trending feed (Aug-2020, US)
├── results
│   ├── results.csv                # per-video predictions & probabilities
│   └── quality_score_model.pkl    # trained Random-Forest classifier
├── src
│   ├── data_discover.py           # (≈ 01_data_discover.py)  API check & filter
│   ├── clean_and_aggregate.py     # (≈ 02_clean_and_aggregate.py)  clean comments
│   ├── preprocessing.py           # (≈ 03_comments_only_prep.py)  label & transform
│   ├── model_training.py          # (≈ 04_train_rf_classifier.py)  RF training
│   └── predict_score.py           # batch scoring with trained model
├── requirements.txt
└── README.md
```

---

1.  Quick start
    -------------
    ```bash
    # 1. Add YouTube Data API key
    export YOUTUBE_API_KEY="YOUR_KEY"

    # 2. Run full pipeline
    python src/data_discover.py
    python src/clean_and_aggregate.py
    python src/preprocessing.py
    python src/model_training.py
    ```

2.  What each script does
    ----------------------
    | Script | Purpose | Key outputs |
    |--------|---------|-------------|
    | `data_discover.py` | Filters the raw `data/data.csv` to videos whose comments are publicly accessible via the YouTube API. | `results/videos_with_comments_enabled.csv` |
    | `clean_and_aggregate.py` | Loads comment JSON, removes duplicates, and aggregates **per-video engagement metrics** (creator reply rate, avg comment length, etc.). | `results/video_metrics.csv` |
    | `preprocessing.py` | Keeps videos with ≥1 comment, defines success labels (top 25 % views / likes / likes-per-view), log-transforms interaction features. | `results/video_metrics_final.csv` |
    | `model_training.py` | Trains a Random-Forest classifier on the final features, writes model + metrics. | `results/quality_score_model.pkl`, console report |
    | `predict_score.py` | Loads the trained model and scores any new CSV that contains the same feature columns. | `results/results.csv` |

3.  Configuration
    --------------
    Edit `config/predict.yaml` to change:

    *   Model position
    *   Data reading and output paths
    

    Example:
    ```yaml
    csv:
        input:  "data/data.csv"
        output: "results/data/results.csv"


    model:
        path: "results/metrics/quality_score_model.pkl"
    ```

4.  Reproducing / extending
    -------------------------
    *   **New feed**: drop any YouTube trending CSV into `data/data.csv` (keep same column names).
    *   **Different model**: replace `model_training.py` with your own estimator (e.g., XGBoost). Ensure it serializes to `results/quality_score_model.pkl`.
    *   **Real-time scoring**: expose `predict_score.py` as a REST micro-service (FastAPI template under `extras/` if needed).

5.  Outputs explained
    -------------------
    *   `results.csv`  
      `video_id`, `predicted_success`, `success_prob`, plus original metadata.
    *   `quality_score_model.pkl`  
      A `sklearn.pipeline.Pipeline` object ready for `joblib.load`.
    *   Console logs include **accuracy, precision, recall, F1** and a **feature-importance bar-chart**.

6.  Troubleshooting
    -----------------
    | Issue | Fix |
    |-------|-----|
    | `KeyError: 'YOUTUBE_API_KEY'` | `export YOUTUBE_API_KEY="..."` |
    | `quotaExceeded` | Use a billing-enabled GCP project or reduce `batch_size`. |
    | Unicode warnings | Ensure `PYTHONIOENCODING=utf-8` in your shell. |

## 📄 License

m13052784890@163.com

Feel free to open issues or PRs!