"""
Classification v2: VisitMode prediction with the same leak-safe engineered features.
"""
import pandas as pd, numpy as np, joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import lightgbm as lgb
import xgboost as xgb

m = pd.read_csv('data/cleaned/master_features.csv')
train, test = m[m.__split == 'train'].copy(), m[m.__split == 'test'].copy()

cat_cols = ['UserContinent', 'UserRegion', 'UserCountry', 'AttractionType', 'AttractionRegion', 'User_FavAttractionType']
num_cols = ['VisitYear', 'VisitMonth', 'User_AvgRating', 'User_TotalVisits', 'User_FavMonth',
            'Attraction_AvgRating', 'Attraction_TotalVisits', 'City_AvgRating']

encoders = {}
def encode(df, fit=False):
    out = df[cat_cols + num_cols].copy()
    for c in cat_cols:
        if fit:
            le = LabelEncoder(); le.fit(out[c].astype(str)); encoders[c] = le
        le = encoders[c]
        out[c] = out[c].astype(str).map(lambda v: v if v in le.classes_ else le.classes_[0])
        out[c] = le.transform(out[c])
    return out

X_train, X_test = encode(train, fit=True), encode(test)
y_le = LabelEncoder(); y_le.fit(train['VisitModeLabel'])
y_train = y_le.transform(train['VisitModeLabel'])
y_test = y_le.transform(test['VisitModeLabel'])

scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

models = {}
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train_s, y_train); models['LogisticRegression'] = (logreg, X_test_s)
rf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train); models['RandomForest'] = (rf, X_test)
lgbm = lgb.LGBMClassifier(n_estimators=300, max_depth=6, class_weight='balanced', random_state=42, verbosity=-1)
lgbm.fit(X_train, y_train); models['LightGBM'] = (lgbm, X_test)
sample_weight = pd.Series(y_train).map(pd.Series(y_train).value_counts(normalize=True).rdiv(1)).values
xgbc = xgb.XGBClassifier(n_estimators=300, max_depth=5, random_state=42, eval_metric='mlogloss')
xgbc.fit(X_train, y_train, sample_weight=sample_weight); models['XGBoost'] = (xgbc, X_test)

rows = []
for name, (model, Xte) in models.items():
    pred = model.predict(Xte)
    tr_pred = model.predict(X_train_s if name == 'LogisticRegression' else X_train)
    rows.append({
        'Model': name,
        'Train_Accuracy': accuracy_score(y_train, tr_pred),
        'Test_Accuracy': accuracy_score(y_test, pred),
        'Test_Macro_F1': f1_score(y_test, pred, average='macro'),
        'Test_Macro_Precision': precision_score(y_test, pred, average='macro', zero_division=0),
        'Test_Macro_Recall': recall_score(y_test, pred, average='macro', zero_division=0),
    })

results_df = pd.DataFrame(rows).sort_values('Test_Macro_F1', ascending=False)
print(results_df.to_string(index=False))
results_df.to_csv('models/classification_model_comparison_v2.csv', index=False)

best_name = results_df.iloc[0]['Model']
best_model, best_Xte = models[best_name]
best_pred = best_model.predict(best_Xte)
print(f"\nBest model: {best_name}")
print(classification_report(y_test, best_pred, target_names=y_le.classes_))

joblib.dump({'model': best_model, 'encoders': encoders, 'target_encoder': y_le,
             'cat_cols': cat_cols, 'num_cols': num_cols,
             'scaler': scaler if best_name == 'LogisticRegression' else None, 'model_name': best_name},
            'models/visitmode_classifier.pkl')
print("Saved models/visitmode_classifier.pkl")
