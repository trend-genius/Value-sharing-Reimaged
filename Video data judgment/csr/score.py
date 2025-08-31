import joblib, pandas as pd, numpy as np

# 1. Load model + mapping parameters
model    = joblib.load("tiktok_like_rate_predictor.joblib")
q01, q99 = joblib.load("score_mapping.joblib").values()

# 2. 0-100 score mapping function
def like_rate_to_score(rate):
    return np.clip((rate - q01) / (q99 - q01), 0, 1) * 100

# 3. Load videos to score
df = pd.read_csv("tiktok_dataset.csv")   # or your own file

# 4. Build features
eps = 1e-6
df['comment_rate'] = df['video_comment_count'] / (df['video_view_count'] + eps)
df['share_rate']   = df['video_share_count']  / (df['video_view_count'] + eps)
df['log_comment_rate'] = np.log1p(df['comment_rate'] * 1000)
df['log_share_rate']   = np.log1p(df['share_rate']   * 1000)

features = ['video_duration_sec', 'log_comment_rate', 'log_share_rate']

# 5. Compute 0-100 score
df['score'] = like_rate_to_score(model.predict(df[features]))

# 6. Save results (keep only id + score; add columns as needed)
out = df[['score']].copy()
out.to_csv("video_score_0to100.csv", index_label="video_id")
print("0-100 score results exported to video_score_0to100.csv")