"""
Regression v2: adds leak-safe engineered features on top of the v1 feature set.
Compares against v1 (raw demographic/attraction features only) to check whether
the added features are worth the complexity, not just assumed to help.
"""
import pandas as pd, numpy as np, joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

m = pd.read_csv('data/cleaned/master_features.csv')
train, test = m[m.__split == 'train'].copy(), m[m.__split == 'test'].copy()

cat_cols = ['UserContinent', 'UserRegion', 'UserCountry', 'AttractionType', 'AttractionRegion',
            'VisitModeLabel', 'User_FavAttractionType']
num_cols = ['VisitYear', 'VisitMonth', 'User_AvgRating', 'User_TotalVisits',
            'User_FavMonth', 'Attraction_AvgRating', 'Attraction_TotalVisits', 'City_AvgRating']

encoders = {}
def encode(df, fit=False):
    out = df[cat_cols + num_cols].copy()
    for c in cat_cols:
        if fit:
            le = LabelEncoder(); le.fit(out[c].astype(str))
            encoders[c] = le
        le = encoders[c]
        out[c] = out[c].astype(str).map(lambda v: v if v in le.classes_ else le.classes_[0])
        out[c] = le.transform(out[c])
    return out

X_train = encode(train, fit=True)
X_test = encode(test)
y_train, y_test = train['Rating'].values, test['Rating'].values

rows = []
baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
rows.append({'Model': 'Baseline (mean)', 'R2': r2_score(y_test, baseline_pred),
             'MAE': mean_absolute_error(y_test, baseline_pred),
             'RMSE': np.sqrt(mean_squared_error(y_test, baseline_pred))})

scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
models = {}
lr = LinearRegression(); lr.fit(X_train_s, y_train); models['LinearRegression'] = (lr, X_test_s)
rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train); models['RandomForest'] = (rf, X_test)
lgbm = lgb.LGBMRegressor(n_estimators=300, max_depth=8, random_state=42, verbosity=-1)
lgbm.fit(X_train, y_train); models['LightGBM'] = (lgbm, X_test)
xgbr = xgb.XGBRegressor(n_estimators=300, max_depth=6, random_state=42)
xgbr.fit(X_train, y_train); models['XGBoost'] = (xgbr, X_test)

for name, (model, Xte) in models.items():
    pred = model.predict(Xte)
    rows.append({'Model': name, 'R2': r2_score(y_test, pred), 'MAE': mean_absolute_error(y_test, pred),
                 'RMSE': np.sqrt(mean_squared_error(y_test, pred))})

results_df = pd.DataFrame(rows).sort_values('R2', ascending=False)
print("=== v2 (with engineered features) ===")
print(results_df.to_string(index=False))
results_df.to_csv('models/regression_model_comparison_v2.csv', index=False)

best_name = results_df.iloc[0]['Model']
best_model, best_Xte = models.get(best_name, (lgbm, X_test))
print(f"\nBest model: {best_name}")

joblib.dump({'model': best_model, 'encoders': encoders, 'cat_cols': cat_cols, 'num_cols': num_cols,
             'scaler': scaler if best_name == 'LinearRegression' else None, 'model_name': best_name},
            'models/rating_regressor.pkl')

imp = pd.Series(models['LightGBM'][0].feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nFeature importance (LightGBM v2):\n", imp)
