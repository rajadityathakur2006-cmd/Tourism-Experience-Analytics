import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")

@st.cache_resource
def load_artifacts():
    clf = joblib.load(os.path.join(BASE, 'models/visitmode_classifier.pkl'))
    reg = joblib.load(os.path.join(BASE, 'models/rating_regressor.pkl'))
    rec = joblib.load(os.path.join(BASE, 'models/recommender.pkl'))
    master = pd.read_csv(os.path.join(BASE, 'data/cleaned/master_clean.csv'))
    users = pd.read_csv(os.path.join(BASE, 'data/cleaned/users_clean.csv'))
    attr = pd.read_csv(os.path.join(BASE, 'data/cleaned/attractions_clean.csv'))
    return clf, reg, rec, master, users, attr

clf, reg, rec, master, users, attr = load_artifacts()

st.title("Tourism Experience Analytics")
st.caption("Classification, prediction, and recommendation for attraction visits")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Predict Visit Mode", "Predict Rating", "Recommend Attractions"])

with tab1:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(master):,}")
    c2.metric("Users", f"{master.UserId.nunique():,}")
    c3.metric("Attractions", f"{attr.shape[0]}")

    st.markdown("**Rating distribution**")
    st.bar_chart(master.Rating.value_counts().sort_index())

    st.markdown("**Visit mode distribution**")
    st.bar_chart(master.VisitModeLabel.value_counts())

    st.markdown("**Average rating by attraction type**")
    st.bar_chart(master.groupby('AttractionType').Rating.mean().sort_values(ascending=False))

    st.markdown("**Model comparison — Visit Mode classification**")
    try:
        clf_comp = pd.read_csv(os.path.join(BASE, 'models/classification_model_comparison_v2.csv'))
        st.dataframe(clf_comp, use_container_width=True)
    except FileNotFoundError:
        pass

    st.markdown("**Model comparison — Rating regression**")
    try:
        reg_comp = pd.read_csv(os.path.join(BASE, 'models/regression_model_comparison_v2.csv'))
        st.dataframe(reg_comp, use_container_width=True)
    except FileNotFoundError:
        pass

    st.info(
        "Note: VisitMode classification tops out around 40% accuracy / 0.31 macro-F1 across "
        "4 algorithms tested, and rating regression tops out around R2~0.11. Geography and "
        "attraction-type features don't strongly determine either target - treat model outputs "
        "below as directional, not precise predictions."
    )

    st.markdown("---")
    st.caption(
        "Both prediction models below use engineered features (average rating by this "
        "attraction, by this user, etc.) computed with K-fold out-of-fold encoding. A "
        "first-time visitor to this app has no rating history, so 'User' features "
        "default to dataset-wide averages - identical to how new/cold-start users were "
        "handled during training."
    )

# --- Lookup tables built once from historical data, used by both predict tabs ---
attr_stats = master.groupby('AttractionId').agg(
    Attraction_AvgRating=('Rating', 'mean'), Attraction_TotalVisits=('Rating', 'size')).reset_index()
attr_lookup = attr.merge(attr_stats, on='AttractionId', how='left')
city_stats = master.groupby('UserCity')['Rating'].mean().rename('City_AvgRating')
GLOBAL_MEAN_RATING = master['Rating'].mean()

with tab2:
    st.subheader("Predict likely visit mode")
    col1, col2 = st.columns(2)
    with col1:
        continent = st.selectbox("User continent", sorted(users.UserContinent.dropna().unique()))
        region_opt = st.selectbox("User region", sorted(users.UserRegion.dropna().unique()))
        country_opt = st.selectbox("User country", sorted(users.UserCountry.dropna().unique()))
        city_opt = st.selectbox("User city", sorted(users.UserCity.dropna().unique()))
    with col2:
        attraction_choice = st.selectbox("Attraction", sorted(attr_lookup.Attraction.unique()))
        visit_month = st.slider("Visit month", 1, 12, 6)
        visit_year = st.slider("Visit year", 2013, 2026, 2024)
        new_user = st.checkbox("This is a first-time visitor (no rating history)", value=True, key='nu1')

    if st.button("Predict visit mode"):
        a_row = attr_lookup[attr_lookup.Attraction == attraction_choice].iloc[0]
        row = {
            'UserContinent': continent, 'UserRegion': region_opt, 'UserCountry': country_opt,
            'AttractionType': a_row['AttractionType'], 'AttractionRegion': a_row['AttractionRegion'],
            'VisitYear': visit_year, 'VisitMonth': visit_month,
            'User_AvgRating': GLOBAL_MEAN_RATING, 'User_TotalVisits': 0,
            'User_FavAttractionType': 'Unknown', 'User_FavMonth': 0,
            'Attraction_AvgRating': a_row['Attraction_AvgRating'],
            'Attraction_TotalVisits': a_row['Attraction_TotalVisits'],
            'City_AvgRating': city_stats.get(city_opt, GLOBAL_MEAN_RATING),
        }
        X = pd.DataFrame([row])
        for c in clf['cat_cols']:
            le = clf['encoders'][c]
            X[c] = X[c].map(lambda v: v if v in le.classes_ else le.classes_[0])
            X[c] = le.transform(X[c].astype(str))
        X = X[clf['cat_cols'] + clf['num_cols']]
        if clf.get('scaler') is not None:
            X = clf['scaler'].transform(X)
        pred = clf['model'].predict(X)[0]
        proba = clf['model'].predict_proba(X)[0]
        label = clf['target_encoder'].inverse_transform([pred])[0]
        st.success(f"Predicted visit mode: **{label}**")
        proba_df = pd.Series(proba, index=clf['target_encoder'].classes_).sort_values(ascending=False)
        st.bar_chart(proba_df)
        st.caption(f"Model: {clf.get('model_name', 'n/a')} — macro-F1 ~0.31 on held-out test data. Treat as directional.")

with tab3:
    st.subheader("Predict attraction rating")
    col1, col2 = st.columns(2)
    with col1:
        continent2 = st.selectbox("User continent ", sorted(users.UserContinent.dropna().unique()), key='c2')
        region2 = st.selectbox("User region ", sorted(users.UserRegion.dropna().unique()), key='r2')
        country2 = st.selectbox("User country ", sorted(users.UserCountry.dropna().unique()), key='co2')
        city2 = st.selectbox("User city ", sorted(users.UserCity.dropna().unique()), key='ci2')
    with col2:
        attraction_choice2 = st.selectbox("Attraction ", sorted(attr_lookup.Attraction.unique()), key='at2')
        visit_mode2 = st.selectbox("Visit mode", sorted(master.VisitModeLabel.unique()))
        visit_month2 = st.slider("Visit month ", 1, 12, 6, key='m2')
        visit_year2 = st.slider("Visit year ", 2013, 2026, 2024, key='y2')

    if st.button("Predict rating"):
        a_row2 = attr_lookup[attr_lookup.Attraction == attraction_choice2].iloc[0]
        row = {
            'UserContinent': continent2, 'UserRegion': region2, 'UserCountry': country2,
            'AttractionType': a_row2['AttractionType'], 'AttractionRegion': a_row2['AttractionRegion'],
            'VisitModeLabel': visit_mode2, 'VisitYear': visit_year2, 'VisitMonth': visit_month2,
            'User_AvgRating': GLOBAL_MEAN_RATING, 'User_TotalVisits': 0,
            'User_FavAttractionType': 'Unknown', 'User_FavMonth': 0,
            'Attraction_AvgRating': a_row2['Attraction_AvgRating'],
            'Attraction_TotalVisits': a_row2['Attraction_TotalVisits'],
            'City_AvgRating': city_stats.get(city2, GLOBAL_MEAN_RATING),
        }
        X = pd.DataFrame([row])
        for c in reg['cat_cols']:
            le = reg['encoders'][c]
            X[c] = X[c].map(lambda v: v if v in le.classes_ else le.classes_[0])
            X[c] = le.transform(X[c].astype(str))
        X = X[reg['cat_cols'] + reg['num_cols']]
        if reg.get('scaler') is not None:
            X = reg['scaler'].transform(X)
        pred = reg['model'].predict(X)[0]
        st.success(f"Predicted rating: **{pred:.2f} / 5**")
        st.caption(f"Model: {reg.get('model_name', 'n/a')} — R² ~0.16 on held-out test data (up from 0.11 pre-feature-engineering). Directional, not precise.")

with tab4:
    st.subheader("Recommended attractions")
    user_input = st.text_input("Enter a UserId (leave blank for a new/cold-start user)")

    def recommend_for_user(user_id, top_n=5, cf_weight=0.7, content_weight=0.3):
        ui = rec['ui']; item_sim_cf = rec['item_sim_cf']; item_sim_content = rec['item_sim_content']
        popularity_rank = rec['popularity_rank']
        user_ratings = ui.loc[user_id].dropna() if user_id in ui.index else pd.Series(dtype=float)

        if len(user_ratings) == 0:
            return popularity_rank.head(top_n).index.tolist(), 'Popular attractions (no history for this user)'

        if len(user_ratings) == 1:
            liked_item = user_ratings.index[0]
            sims = item_sim_content[liked_item].drop(index=liked_item, errors='ignore')
            return sims.sort_values(ascending=False).head(top_n).index.tolist(), 'Similar to the one attraction you rated (content-based)'

        cf_scores = pd.Series(0.0, index=ui.columns); cf_weight_sum = pd.Series(0.0, index=ui.columns)
        content_scores = pd.Series(0.0, index=ui.columns)
        for item, rating in user_ratings.items():
            cf_sims = item_sim_cf[item]
            cf_scores += cf_sims * rating
            cf_weight_sum += cf_sims.abs()
            content_scores += item_sim_content[item] * rating
        cf_weight_sum = cf_weight_sum.replace(0, np.nan)
        cf_final = (cf_scores / cf_weight_sum).fillna(0)
        content_final = content_scores / len(user_ratings)
        cf_norm = (cf_final - cf_final.min()) / (cf_final.max() - cf_final.min() + 1e-9)
        content_norm = (content_final - content_final.min()) / (content_final.max() - content_final.min() + 1e-9)
        hybrid = (cf_weight * cf_norm + content_weight * content_norm).drop(index=user_ratings.index, errors='ignore')
        return hybrid.sort_values(ascending=False).head(top_n).index.tolist(), 'Based on your rating history (hybrid collaborative + content filtering)'

    if st.button("Get recommendations"):
        try:
            uid = int(user_input) if user_input.strip() else -1
        except ValueError:
            uid = -1
        recs, method = recommend_for_user(uid)
        st.caption(method)
        rec_df = attr.set_index('AttractionId').loc[recs, ['Attraction', 'AttractionType', 'AttractionRegion']]
        st.table(rec_df.reset_index(drop=True))
