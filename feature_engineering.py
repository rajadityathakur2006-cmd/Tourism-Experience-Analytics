"""
Feature engineering with K-FOLD out-of-fold target encoding.

Why not plain leave-one-out: a first attempt using LOO ((group_sum - own_rating)
/ (group_count - 1)) technically excludes each row's own value, but with
Rating restricted to only 5 discrete values and group sizes known, that
formula is close to invertible - a tree model could reconstruct a row's own
rating from the tiny numeric fingerprint the subtraction leaves behind.
Result: train R2 of 0.50 that collapsed to test R2 of 0.007 - a textbook
overfitting signature, confirmed by checking that the true achievable signal
from attraction-grouping alone is R2 ~0.09 on BOTH train and test (verified
with a plain groupby-mean lookup).

Fix: 5-fold out-of-fold encoding. Each training row's aggregate is computed
using only the OTHER 4 folds (never a formula involving its own value), which
removes the invertibility. Test rows use the full train-set aggregate,
computed once (safe, since test was never part of that computation at all).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

m = pd.read_csv('data/cleaned/master_clean.csv').sort_values('TransactionId').reset_index(drop=True)
train_idx, test_idx = train_test_split(m.index, test_size=0.2, random_state=42)
m['__split'] = np.where(m.index.isin(train_idx), 'train', 'test')

GLOBAL_MEAN = m.loc[m.__split == 'train', 'Rating'].mean()
N_SPLITS = 5

def kfold_target_encode(df, group_col, value_col='Rating', n_splits=N_SPLITS, smoothing=10):
    """Out-of-fold mean encoding with Bayesian smoothing toward the global mean
    (protects small groups from being estimated off too few points)."""
    result = pd.Series(index=df.index, dtype=float)
    train_mask = df.__split == 'train'
    train_df = df[train_mask]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fit_pos, holdout_pos in kf.split(train_df):
        fit_idx = train_df.index[fit_pos]
        holdout_idx = train_df.index[holdout_pos]
        stats = df.loc[fit_idx].groupby(group_col)[value_col].agg(['mean', 'count'])
        smoothed = (stats['mean'] * stats['count'] + GLOBAL_MEAN * smoothing) / (stats['count'] + smoothing)
        result.loc[holdout_idx] = df.loc[holdout_idx, group_col].map(smoothed).fillna(GLOBAL_MEAN).values

    # test rows: full train-set stats (safe - test was never involved)
    full_stats = train_df.groupby(group_col)[value_col].agg(['mean', 'count'])
    full_smoothed = (full_stats['mean'] * full_stats['count'] + GLOBAL_MEAN * smoothing) / (full_stats['count'] + smoothing)
    test_idx_ = df.index[~train_mask]
    result.loc[test_idx_] = df.loc[test_idx_, group_col].map(full_smoothed).fillna(GLOBAL_MEAN).values
    return result

def kfold_count_encode(df, group_col, n_splits=N_SPLITS):
    result = pd.Series(index=df.index, dtype=float)
    train_mask = df.__split == 'train'
    train_df = df[train_mask]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fit_pos, holdout_pos in kf.split(train_df):
        fit_idx = train_df.index[fit_pos]
        holdout_idx = train_df.index[holdout_pos]
        counts = df.loc[fit_idx].groupby(group_col).size()
        result.loc[holdout_idx] = df.loc[holdout_idx, group_col].map(counts).fillna(0).values
    full_counts = train_df.groupby(group_col).size()
    test_idx_ = df.index[~train_mask]
    result.loc[test_idx_] = df.loc[test_idx_, group_col].map(full_counts).fillna(0).values
    return result

def kfold_mode_encode(df, group_col, value_col, n_splits=N_SPLITS):
    result = pd.Series(index=df.index, dtype=object)
    train_mask = df.__split == 'train'
    train_df = df[train_mask]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fit_pos, holdout_pos in kf.split(train_df):
        fit_idx = train_df.index[fit_pos]
        holdout_idx = train_df.index[holdout_pos]
        modes = df.loc[fit_idx].groupby(group_col)[value_col].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else 'Unknown')
        result.loc[holdout_idx] = df.loc[holdout_idx, group_col].map(modes).fillna('Unknown').values
    full_modes = train_df.groupby(group_col)[value_col].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else 'Unknown')
    test_idx_ = df.index[~train_mask]
    result.loc[test_idx_] = df.loc[test_idx_, group_col].map(full_modes).fillna('Unknown').values
    return result

m['User_AvgRating'] = kfold_target_encode(m, 'UserId', smoothing=3)
m['User_TotalVisits'] = kfold_count_encode(m, 'UserId')
m['User_FavAttractionType'] = kfold_mode_encode(m, 'UserId', 'AttractionType')
m['User_FavMonth'] = m.groupby('UserId')['VisitMonth'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else 0)

m['Attraction_AvgRating'] = kfold_target_encode(m, 'AttractionId', smoothing=20)
m['Attraction_TotalVisits'] = kfold_count_encode(m, 'AttractionId')

m['City_AvgRating'] = kfold_target_encode(m, 'UserCity', smoothing=10)

m.to_csv('data/cleaned/master_features.csv', index=False)
print("Saved. Shape:", m.shape)

# --- Verification: can a model reconstruct a row's own rating from the encoded feature alone? ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
train, test = m[m.__split == 'train'], m[m.__split == 'test']
rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(train[['Attraction_AvgRating']], train['Rating'])
tr_r2 = r2_score(train['Rating'], rf.predict(train[['Attraction_AvgRating']]))
te_r2 = r2_score(test['Rating'], rf.predict(test[['Attraction_AvgRating']]))
print(f"\nLeakage re-check (Attraction_AvgRating only, RF depth=5): train_R2={tr_r2:.3f}  test_R2={te_r2:.3f}")
print("(train and test should now be close together, unlike the 0.505 / 0.007 seen with plain LOO)")
