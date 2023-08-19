import numpy as np
import implicit
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, csr_matrix

from src_v_2_pandas.classes.utils import *
from src_v_2_pandas.classes.paths_config import *
from src_v_2_pandas.classes.ALSRecommender import ALSRecommender

all_transactions = pd.read_parquet( Path( raw_dir, "transactions_train.parquet" ),
                                    columns=["t_dat", "customer_id", "article_id"] )
all_transactions["t_dat"] = pd.to_datetime( all_transactions["t_dat"] )
#print( all_transactions.shape )
#all_transactions = all_transactions[all_transactions['t_dat'] > '2020-06-21']
#print( all_transactions.shape )
#all_transactions.to_parquet( Path( raw_dir, "transactions_train_short.parquet" ) )
#all_transactions = pd.read_parquet( Path( raw_dir, "transactions_train_short.parquet" ) )

all_users_df = pd.read_parquet( Path( raw_dir, "customers.parquet" ), columns=["customer_id"] )
all_items_df = pd.read_parquet( Path( raw_dir, "articles.parquet" ), columns=["article_id"] )

als_recommender = ALSRecommender( all_users_df, all_items_df )
all_transactions = als_recommender.add_mapped_user_item_ids( all_transactions )

#als_recommender.evaluate_als_cv( all_transactions, cv=5, train_days=28, validation_days=7,
#                                factors=100, iterations=12,
#                                nlist=400, nprobe=20, use_gpu=False,
#                                calculate_training_loss=False,
#                                regularization=0.01, random_state=45 )

als_recommender.fit_als(all_transactions,
                        factors=500, iterations=3,
                        nlist=400, nprobe=20, use_gpu=False,
                        calculate_training_loss=False,
                        regularization=0.01, random_state=45 )
save( als_recommender, Path(interim_dir, "als_recommender.pkl") )


als_recommender = load(Path(interim_dir, "als_recommender.pkl"))
sample_submission = pd.read_csv( Path(raw_dir, "sample_submission.csv") )
user_ids = sample_submission["customer_id"].tolist()

###################
"""all_transactions = pd.read_parquet( Path( raw_dir, "transactions_train.parquet" ), columns=["customer_id"] )
users_with_transactions = all_transactions["customer_id"].tolist()
users_with_transactions = set( users_with_transactions )

users_without_transactions = set(user_ids)
users_without_transactions = users_without_transactions.difference( users_with_transactions )

users_with_transactions = list(users_with_transactions)
users_without_transactions = list(users_without_transactions)

als_recommendations_without_trans = als_recommender.recommend(users_without_transactions, N=12, filter_already_liked_items=False, batch_size=100)
als_recommendations = als_recommender.recommend(users_with_transactions, N=12, filter_already_liked_items=False, batch_size=2000)"""
###################

als_recommendations = als_recommender.recommend(user_ids, N=12, filter_already_liked_items=False, batch_size=2000)
save( als_recommendations, Path( interim_dir, "als_recommendations.pkl" ) )

print("done")