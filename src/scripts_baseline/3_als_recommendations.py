import numpy as np
import implicit
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import csr_matrix, csr_matrix

from src_v_1_no_cv.classes.utils import *
from src_v_1_no_cv.classes.paths_config import *
from src_v_1_no_cv.classes.ALSRecommender import ALSRecommender

frequency_dict = load(Path(interim_dir, "frequency_dict.pkl"))

"""keys = list(frequency_dict.keys())[:1000]
subsample_dict = {}
for key in keys:
   subsample_dict[key] = frequency_dict[key]
save( subsample_dict, Path(interim_dir, "subsample_dict.pkl") )
frequency_dict = load( Path(interim_dir, "subsample_dict.pkl") )"""

users_with_transactions = []
users_without_transactions = []
for user_id in frequency_dict.keys():
    if len(frequency_dict[user_id]) == 0:
        users_without_transactions.append( user_id )
        del frequency_dict[user_id]
    else:
        users_with_transactions.append( user_id )
save( users_with_transactions, Path(interim_dir, "users_with_transactions.pkl") )
save( users_without_transactions, Path(interim_dir, "users_without_transactions.pkl") )
clean_frequency_dict = frequency_dict

als_recommender = ALSRecommender()
csr_u2i_matrix = als_recommender.build_csr_u2i_matrix( clean_frequency_dict, use_sum_counts=False, n_most_popular=12 )
save(csr_u2i_matrix, Path(interim_dir, "csr_u2i_matrix.pkl"))
save(als_recommender, Path(interim_dir, "als_recommender.pkl"))

csr_u2i_matrix = load( Path(interim_dir, "csr_u2i_matrix.pkl") )
als_recommender.fit_als(csr_u2i_matrix,
                factors=500, iterations=3,
                nlist=400, nprobe=20, use_gpu=False,
                calculate_training_loss=False,
                regularization=0.01, random_state=45)
save(als_recommender, Path(interim_dir, "als_recommender.pkl"))

"""users_without_transactions = load( Path(interim_dir, "users_without_transactions.pkl") )
users_with_transactions = load( Path(interim_dir, "users_with_transactions.pkl") )
als_recommender = load( Path(interim_dir, "als_recommender.pkl") )
recommends_for_uwot = als_recommender.recommend( users_without_transactions, N=12, filter_already_liked_items=False )
recommends_for_uwit = als_recommender.recommend( users_with_transactions, N=12, filter_already_liked_items=False )
als_recommendations = recommends_for_uwot | recommends_for_uwit
save( als_recommendations, Path( interim_dir, "als_recommendations.pkl" ) )"""

als_recommender = load( Path(interim_dir, "als_recommender.pkl") )
sample_submission = pd.read_csv( Path(raw_dir, "sample_submission.csv") )
user_ids = sample_submission["customer_id"].tolist()
als_recommendations = als_recommender.recommend( user_ids, N=12, filter_already_liked_items=False )
save( als_recommendations, Path( interim_dir, "als_recommendations.pkl" ) )

"""als_recommender = load(Path(interim_dir, "als_recommender.pkl"))
user_factors = als_recommender.get_user_factors()
article_factors = als_recommender.get_item_factors()

factor_dict_postfix = "50_100_400_20_1"
article_factor_features_dict = {}
article_factor_features_dict["feature_names"] = []
for i in range(len(article_factors[0])):
    article_factor_features_dict["feature_names"].append("factor_{}".format(i))
inverted_article_id_dict = load(Path(interim_dir, "factor_inverted_article_id_dict.pkl"))
for i in tqdm(range(len(article_factors)), desc="Building article factor features dict"):
    article_id = inverted_article_id_dict[i]
    article_factor_features = article_factors[i]
    article_factor_features_dict[article_id] = article_factor_features
save(article_factor_features_dict, Path(interim_dir, "article_factor_features_dict_{}.pkl".format(factor_dict_postfix)))

user_factor_features_dict = {}
user_factor_features_dict["feature_names"] = []
for i in range(len(user_factors[0])):
    user_factor_features_dict["feature_names"].append("factor_{}".format(i))

inverted_user_id_dict = load(Path(interim_dir, "factor_inverted_user_id_dict.pkl"))
for i in tqdm(range(len(user_factors)), desc="Building user factor features dict"):
    user_id = inverted_user_id_dict[i]
    user_factor_features = user_factors[i]
    user_factor_features_dict[user_id] = user_factor_features
save(user_factor_features_dict, Path(interim_dir, "user_factor_features_dict_{}.pkl".format(factor_dict_postfix)))"""

print("done")