import argparse
import yaml
import joblib
import pandas as pd
from pathlib import Path

# ---------- 1. Locate repository root ----------
# src/predict_csv.py -> project_root
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------- 2. Load YAML config ----------
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config",
    default="config/predict.yaml",
    help="YAML config file (relative to repository root)"
)
args = parser.parse_args()

cfg_path = REPO_ROOT / args.config          # absolute path
with cfg_path.open("rt", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# ---------- 3. Convert relative paths in YAML to absolute ----------
csv_in  = REPO_ROOT / cfg["csv"]["input"]
csv_out = REPO_ROOT / cfg["csv"]["output"]
model_p = REPO_ROOT / cfg["model"]["path"]

# ---------- 4. Main workflow ----------
model = joblib.load(model_p)
df = pd.read_csv(csv_in)

missing = set(cfg["columns"]) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df["score"] = model.predict(df[cfg["columns"]])

csv_out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(csv_out, index=False)

print("✅ Prediction complete; results saved to:", csv_out.relative_to(REPO_ROOT))
print(df[["score"]].round(1))