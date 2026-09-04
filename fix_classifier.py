"""
Run this from inside your Tourism_Project folder:
    python3 fix_classifier.py

Replaces models/visitmode_classifier.pkl with a RandomForest version
(macro-F1 0.293 vs LightGBM's 0.312 - a small drop, but RF has no
system-library dependency, so it sidesteps the libomp/admin-rights issue
entirely). Uses scikit-learn only, nothing that needs sudo.
"""
import pandas as pd, joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

m = pd.read_csv('data/cleaned/master_features.csv')
train, test = m[m.__split == 'train'].copy(), m[m.__split == 'test'].copy()

cat_cols = ['UserContinent', 'UserRegion', 'UserCountry', 'AttractionType',
            'AttractionRegion', 'User_FavAttractionType']
num_cols = ['VisitYear', 'VisitMonth', 'User_AvgRating', 'User_TotalVisits',
            'User_FavMonth', 'Attraction_AvgRating', 'Attraction_TotalVisits', 'City_AvgRating']

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

X_train = encode(train, fit=True)
y_le = LabelEncoder(); y_le.fit(train['VisitModeLabel'])
y_train = y_le.transform(train['VisitModeLabel'])

rf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced',
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

joblib.dump({
    'model': rf, 'encoders': encoders, 'target_encoder': y_le,
    'cat_cols': cat_cols, 'num_cols': num_cols, 'scaler': None, 'model_name': 'RandomForest'
}, 'models/visitmode_classifier.pkl')

print("Done. models/visitmode_classifier.pkl now uses RandomForest - no LightGBM/libomp needed.")
print("Now just run: streamlit run streamlit_app/streamlit_app.py")
