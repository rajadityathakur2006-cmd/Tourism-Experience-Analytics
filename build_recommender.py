"""
Hybrid recommendation system:
 1. Collaborative filtering (item-based cosine similarity) - used when the
    user has 2+ ratings to build a preference signal from.
 2. Content-based filtering (attraction-type similarity) - used to diversify
    / support CF recommendations and as a secondary signal for users with
    exactly 1 rating (recommend attractions similar in TYPE to the one thing
    they've rated, rather than pure popularity).
 3. Popularity fallback (Bayesian-adjusted) - true cold start, zero history.
Reality check: 33,530 users, 30 attractions, 68% of users have exactly one
rating. Pure CF has nothing to work with for most users, so the fallback
tiers are not optional extras - they're what most users actually get.
"""
import pandas as pd, numpy as np, joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

m = pd.read_csv('data/cleaned/master_clean.csv')
attr = pd.read_csv('data/cleaned/attractions_clean.csv')

# --- Collaborative filtering: item-item cosine similarity on ratings ---
ui = m.pivot_table(index='UserId', columns='AttractionId', values='Rating', aggfunc='mean')
ui_filled = ui.fillna(0)
item_sim_cf = pd.DataFrame(cosine_similarity(ui_filled.T), index=ui.columns, columns=ui.columns)

# --- Content-based: attraction similarity from type + region (one-hot) ---
ohe = OneHotEncoder()
content_features = ohe.fit_transform(attr[['AttractionType', 'AttractionRegion']]).toarray()
item_sim_content = pd.DataFrame(cosine_similarity(content_features),
                                 index=attr['AttractionId'], columns=attr['AttractionId'])

# --- Popularity fallback (Bayesian-adjusted average rating) ---
agg = m.groupby('AttractionId').agg(avg_rating=('Rating', 'mean'), n=('Rating', 'size'))
C = agg.n.mean(); mprior = agg.avg_rating.mean()
agg['bayes_score'] = (agg.n / (agg.n + C)) * agg.avg_rating + (C / (agg.n + C)) * mprior
popularity_rank = agg.sort_values('bayes_score', ascending=False)

def recommend_for_user(user_id, top_n=5, cf_weight=0.7, content_weight=0.3):
    if user_id in ui.index:
        user_ratings = ui.loc[user_id].dropna()
    else:
        user_ratings = pd.Series(dtype=float)

    if len(user_ratings) == 0:
        return popularity_rank.head(top_n).index.tolist(), 'cold_start_popularity'

    if len(user_ratings) == 1:
        # single data point: CF has almost no signal, lean on content similarity
        liked_item = user_ratings.index[0]
        sims = item_sim_content[liked_item].drop(index=liked_item, errors='ignore')
        recs = sims.sort_values(ascending=False).head(top_n).index.tolist()
        return recs, 'content_based_single_rating'

    # hybrid score: blend CF and content similarity, weighted toward CF
    cf_scores = pd.Series(0.0, index=ui.columns)
    cf_weight_sum = pd.Series(0.0, index=ui.columns)
    content_scores = pd.Series(0.0, index=ui.columns)
    for item, rating in user_ratings.items():
        cf_sims = item_sim_cf[item]
        cf_scores += cf_sims * rating
        cf_weight_sum += cf_sims.abs()
        content_scores += item_sim_content[item] * rating

    cf_weight_sum = cf_weight_sum.replace(0, np.nan)
    cf_final = (cf_scores / cf_weight_sum).fillna(0)
    content_final = content_scores / len(user_ratings)

    # normalize both to 0-1 before blending
    cf_norm = (cf_final - cf_final.min()) / (cf_final.max() - cf_final.min() + 1e-9)
    content_norm = (content_final - content_final.min()) / (content_final.max() - content_final.min() + 1e-9)
    hybrid = cf_weight * cf_norm + content_weight * content_norm
    hybrid = hybrid.drop(index=user_ratings.index, errors='ignore')
    recs = hybrid.sort_values(ascending=False).head(top_n).index.tolist()
    return recs, 'hybrid_cf_content'

# sanity checks across all 3 tiers
for uid in [70456, m[m.UserId.isin(ui[ui.notna().sum(axis=1)==1].index)].UserId.iloc[0], 999999999]:
    recs, method = recommend_for_user(uid, 5)
    names = attr.set_index('AttractionId').loc[recs, 'Attraction'].tolist()
    print(f"User {uid} ({method}): {names}")

joblib.dump({
    'item_sim_cf': item_sim_cf,
    'item_sim_content': item_sim_content,
    'ui': ui,
    'popularity_rank': popularity_rank,
    'attr': attr
}, 'models/recommender.pkl')
print("\nSaved models/recommender.pkl (hybrid CF + content + popularity)")
