
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.classes.utils import *
from src.classes.paths_config import *

import sys
import logging
logging.basicConfig(level=logging.INFO, filename=Path(log_dir, "als_submission.log"), filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.info( "Loading ALS recommender" )
als_recommender = load(Path(interim_dir, "als_recommender.pkl"))

logging.info( "Loading sample submission" )
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

logging.info( "Making recommendations" )
als_recommendations = als_recommender.recommend(user_ids, N=12, filter_already_liked_items=False, batch_size=2000)

user_ids = []
predicts = []
for user_id in tqdm(als_recommendations.keys(), desc="Building predicts"):
    user_ids.append( user_id )
    user_items_predict = " ".join( als_recommendations[user_id] )
    predicts.append( user_items_predict )

logging.info( "Building submission dataframe" )
submission_df = pd.DataFrame( data=[user_ids, predicts] )
submission_df = submission_df.transpose()
submission_df.columns = ["customer_id", "prediction"]

logging.info( "Saving submission to csv" )
submission_df.to_csv( Path(result_dir, "submission.csv"), index=False )

logging.info( "Done building submission" )