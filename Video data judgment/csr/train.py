import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import matplotlib as mpl

# Font & minus sign settings (use an English-friendly fallback stack)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.float_format', '{:.4f}'.format)

print("Starting TikTok video quality score (0-100) training script...")

# -----------------------------
# 1. Load data
# -----------------------------
data_file = 'tiktok_dataset.csv'
df = pd.read_csv(data_file)

# -----------------------------
# 2. Feature engineering
# -----------------------------
epsilon = 1e-6
df['like_rate']   = df['video_like_count']   / (df['video_view_count'] + epsilon)
df['comment_rate'] = df['video_comment_count'] / (df['video_view_count'] + epsilon)
df['share_rate']  = df['video_share_count']  / (df['video_view_count'] + epsilon)

df['log_comment_rate'] = np.log1p(df['comment_rate'] * 1000)
df['log_share_rate']   = np.log1p(df['share_rate'] * 1000)

features = ['video_duration_sec', 'log_comment_rate', 'log_share_rate']
X = df[features]
y = df['like_rate']

# -----------------------------
# 3. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
print(f"\nData split: {len(X_train)} train samples, {len(X_test)} test samples")

# -----------------------------
# 4. Train model (still predicting like_rate)
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# 5. Compute 0-100 mapping parameters
# -----------------------------
q01, q99 = np.percentile(y_train, [1, 99])
print(f"\n[Mapping params] 1st percentile={q01:.6f}, 99th percentile={q99:.6f}")


def like_rate_to_score(rate, q01=q01, q99=q99):
    """Map like_rate to a 0-100 quality score."""
    return np.clip((rate - q01) / (q99 - q01), 0, 1) * 100


# Convert both ground truth and predictions to 0-100 scale
y_test_score = like_rate_to_score(y_test)
y_pred_score = like_rate_to_score(y_pred)

# -----------------------------
# 6. Evaluation
# -----------------------------
mae_rate  = mean_absolute_error(y_test, y_pred)
r2_rate   = r2_score(y_test, y_pred)

mae_score = mean_absolute_error(y_test_score, y_pred_score)
r2_score_100 = r2_score(y_test_score, y_pred_score)

print("\n=== Evaluation results ===")
print(f"Like-rate MAE : {mae_rate:.6f}")
print(f"Like-rate R²  : {r2_rate:.4f}")
print(f"0-100 score MAE: {mae_score:.2f} points")
print(f"0-100 score R² : {r2_score_100:.4f}")

# -----------------------------
# 7. Feature importance
# -----------------------------
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n=== Feature importance ===")
print(feature_importance)

# -----------------------------
# 8. Visualization
# -----------------------------
plt.figure(figsize=(15, 10))

# Subplot 1: actual vs predicted 0-100 scores
plt.subplot(2, 2, 1)
plt.scatter(y_test_score, y_pred_score, alpha=0.7, s=50)
min_max = [0, 100]
plt.plot(min_max, min_max, 'r--', lw=2)
plt.xlabel('Actual score (0-100)')
plt.ylabel('Predicted score (0-100)')
plt.title('Actual vs Predicted Score (0-100)')
plt.grid(True, alpha=0.3)

# Subplot 2: feature importance
plt.subplot(2, 2, 2)
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance score')
plt.title('Feature Importance')
plt.grid(True, alpha=0.3)

# Subplot 3: comment rate vs like rate
plt.subplot(2, 2, 3)
plt.scatter(df['comment_rate'], df['like_rate'], alpha=0.7, s=20)
plt.xlabel('Comment rate')
plt.ylabel('Like rate')
plt.title('Comment Rate vs Like Rate')
plt.grid(True, alpha=0.3)

# Subplot 4: 0-100 score error distribution
plt.subplot(2, 2, 4)
score_errors = y_pred_score - y_test_score
plt.hist(score_errors, bins=15, alpha=0.7, edgecolor='black')
plt.xlabel('Error (predicted - actual)')
plt.ylabel('Frequency')
plt.title('0-100 Score Error Distribution')
plt.axvline(0, color='r', ls='--')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('like_rate_prediction_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'like_rate_prediction_results.png'")

# -----------------------------
# 9. Save model & mapping
# -----------------------------
joblib.dump(model, 'tiktok_like_rate_predictor.joblib')
joblib.dump({'features': features, 'preprocessing': 'log_rate'}, 'model_features.joblib')
joblib.dump({'q01': q01, 'q99': q99}, 'score_mapping.joblib')
print("\nModel, features, and mapping parameters saved!")

# -----------------------------
# 10. Single-sample inference demo
# -----------------------------
sample = {
    'video_duration_sec': 45,
    'video_view_count': 100_000,
    'video_share_count': 5_000,
    'video_comment_count': 3_000
}
sample_df = pd.DataFrame([sample])
sample_df['comment_rate'] = sample_df['video_comment_count'] / (sample_df['video_view_count'] + epsilon)
sample_df['share_rate']   = sample_df['video_share_count']   / (sample_df['video_view_count'] + epsilon)
sample_df['log_comment_rate'] = np.log1p(sample_df['comment_rate'] * 1000)
sample_df['log_share_rate']   = np.log1p(sample_df['share_rate'] * 1000)

like_rate_pred = model.predict(sample_df[features])[0]
score_pred = like_rate_to_score(like_rate_pred)

print("\n=== Single-sample inference ===")
print(f"Input features: {sample}")
print(f"Predicted like rate: {like_rate_pred:.4f}")
print(f"Predicted video quality score: {score_pred:.1f}/100")
print("=== Script finished ===")