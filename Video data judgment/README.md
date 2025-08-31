README
========

This repository contains a lightweight **TikTok video-quality scorer** that maps any short-form video to a 0–100 quality score.  
The score is derived from three easily-extracted engagement signals:

- `video_duration_sec`  
- `video_comment_count`  
- `video_share_count`  
- `video_view_count` (used only to compute rates)

The model is a **Random-Forest Regressor** trained on real-world data and translated to a 0–100 scale via percentile mapping.

---

1. 📁 Project Layout
-----------------
```
Video data judgment/
├── csr
│   ├── score.py                 # Inference script
│   └── train.py                 # Model training script
├── data
│   └── tiktok_dataset.csv       # Sample dataset
├── model
│   ├── model_features.joblib    # Feature information
│   ├── score_mapping.joblib     # Mapping parameters
└── tiktok_like_rate_predictor.joblib  # Model body
├── result
│   └── video_score_0to100.csv   # Inference results
├── README.md                    # Project documentation file
└── video_quality_config.yaml    # Project configuration file
```

---

2. Quick start
--------------

### 2.1 Train the model

```bash
python train.py
```

This will:

- Read `tiktok_dataset.csv`  
- Train the model  
- Print evaluation metrics  
- Save three artefacts:

  - `tiktok_like_rate_predictor.joblib` – the RF model  
  - `score_mapping.joblib` – 1st & 99th percentile for 0-100 scaling  
  - `model_features.joblib` – list of feature names & pre-processing hints

### 2.2 Score new videos

Create a CSV with at least these columns:

```
video_duration_sec,video_view_count,video_comment_count,video_share_count
```

then run

```bash
python score.py
```

A file `video_score_0to100.csv` is generated, containing one row per input video with its predicted quality score (0–100).

---


3. Evaluation snapshot
----------------------

Typical results on a held-out test set:

| Metric | Value |
|--------|-------|
| Like-rate MAE | 0.000 67 |
| Like-rate R² | 0.87 |
| 0-100 score MAE | 3.4 points |
| 0-100 score R² | 0.87 |

---

4. Customization notes
----------------------

- **Feature tweak**: edit `features` list in both scripts.  
- **Threshold tweak**: adjust the 1 % / 99 % percentiles in `train.py` (search for `q01, q99`).  
- **Model swap**: replace `RandomForestRegressor` with any scikit-learn regressor in `train.py`.

---

## 📄 License

m13052784890@163.com

Feel free to open issues or PRs!
