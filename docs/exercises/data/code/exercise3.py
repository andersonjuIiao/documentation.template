"""
Exercise 3 - Preparing Real-World Data for a Neural Network
ANN-DL 2026.2 - Exercise 1 (Data)
Dataset: Kaggle Spaceship Titanic (train.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
NUM_COLS = ["Age"] + SPEND_COLS

# ---------------------------------------------------------------------------
# A - Get to know the data
# ---------------------------------------------------------------------------
df = pd.read_csv(HERE / "train.csv")

print("Shape:", df.shape)
balance = df["Transported"].value_counts(normalize=True)
print("\nClass balance (Transported):")
print(balance)

print("\nColumn dtypes:")
print(df.dtypes)

missing = df.isna().sum().to_frame("missing_count")
missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
print("\nMissing values per column:")
print(missing)

print("\nSpending columns - mean / median / max:")
print(df[SPEND_COLS].agg(["mean", "median", "max"]).T)

# ---------------------------------------------------------------------------
# B - Split before you transform
# ---------------------------------------------------------------------------
X = df.drop(columns=["Transported"])
y = df["Transported"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# ---------------------------------------------------------------------------
# C - Preprocess (fit everything on X_train only)
# ---------------------------------------------------------------------------
X_train = X_train.copy()
X_test = X_test.copy()

# --- C.1 Missing data ---
num_imputer = SimpleImputer(strategy="median")
X_train[NUM_COLS] = num_imputer.fit_transform(X_train[NUM_COLS])
X_test[NUM_COLS] = num_imputer.transform(X_test[NUM_COLS])

cat_imputer = SimpleImputer(strategy="most_frequent")
X_train[CAT_COLS] = cat_imputer.fit_transform(X_train[CAT_COLS])
X_test[CAT_COLS] = cat_imputer.transform(X_test[CAT_COLS])

# --- C.2 Categorical encoding (one-hot) ---
# handle_unknown="ignore": a category seen only at test time becomes an
# all-zero row instead of raising an error.
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
train_cat = encoder.fit_transform(X_train[CAT_COLS])
test_cat = encoder.transform(X_test[CAT_COLS])
cat_feature_names = encoder.get_feature_names_out(CAT_COLS)

train_cat_df = pd.DataFrame(train_cat, columns=cat_feature_names, index=X_train.index)
test_cat_df = pd.DataFrame(test_cat, columns=cat_feature_names, index=X_test.index)

# --- C.3 Feature engineering ---
for split_df in (X_train, X_test):
    split_df["TotalSpend"] = split_df[SPEND_COLS].sum(axis=1)

drop_cols = ["Cabin", "Name", "PassengerId"]
X_train = X_train.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)

# --- C.4 Heavy tails: log1p ---
food_court_before = X_train["FoodCourt"].copy()

for col in SPEND_COLS + ["TotalSpend"]:
    X_train[col] = np.log1p(X_train[col])
    X_test[col] = np.log1p(X_test[col])

food_court_after = X_train["FoodCourt"].copy()

num_feature_cols = NUM_COLS + ["TotalSpend"]
X_train_num = X_train[num_feature_cols].reset_index(drop=True)
X_test_num = X_test[num_feature_cols].reset_index(drop=True)

X_train_full = pd.concat([X_train_num, train_cat_df.reset_index(drop=True)], axis=1)
X_test_full = pd.concat([X_test_num, test_cat_df.reset_index(drop=True)], axis=1)

# --- C.5 Scaling: Standardization, fit on train only ---
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_full), columns=X_train_full.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_full), columns=X_test_full.columns
)

print("\nScaled training set - min/max per column (sample of 5 cols):")
print(X_train_scaled.describe().loc[["min", "max"]].iloc[:, :5])

# ---------------------------------------------------------------------------
# D - Verify and visualize
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(food_court_before, bins=40, color="tab:blue")
axes[0].set_title("FoodCourt - before (raw)")
axes[0].set_xlabel("FoodCourt")
axes[0].set_ylabel("count")

axes[1].hist(food_court_after, bins=40, color="tab:orange")
axes[1].set_title("FoodCourt - after log1p")
axes[1].set_xlabel("log(1 + FoodCourt)")
axes[1].set_ylabel("count")

fig.suptitle("Figure 6 - Heavy-tailed feature before/after preprocessing")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig6_foodcourt.png", dpi=150)
plt.close(fig)
print("\nFigure 6 saved.")

print("\nFinal checks:")
print("NaN in train:", X_train_scaled.isna().sum().sum())
print("NaN in test:", X_test_scaled.isna().sum().sum())
print("Final training feature matrix shape:", X_train_scaled.shape)
print("Final test feature matrix shape:", X_test_scaled.shape)
print(
    "Training set value range: min =", round(X_train_scaled.values.min(), 4),
    "max =", round(X_train_scaled.values.max(), 4),
)
print(
    "Test set value range: min =", round(X_test_scaled.values.min(), 4),
    "max =", round(X_test_scaled.values.max(), 4),
)

print("\nDone.")