
import numpy as np
import implicit
from pathlib import Path

import json
import psycopg2 as pg_driver
import clickhouse_driver
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, csr_matrix

from classes.utils import save, load
from classes.ALSRecommender import ALSRecommender

import sys
import logging
logging.basicConfig(level=logging.INFO, filename=Path(".", "model_retraining.log"), filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

class ALSModelRetrainer():
    def __init__(self, db_driver, db_config, als_config):

        self.db_driver = db_driver
        self.db_config = db_config
        self.als_config = als_config

        pass

    def get_db_cursor_(self):
        db_connect = self.db_driver.connect(**self.db_config)
        db_cursor = db_connect.cursor()
        return db_cursor

    def collect_necessary_data_(self, db_cursor):

        logging.info("Extracting customer ids from DB")
        db_cursor.execute("select customer_id from customers;")
        all_users_ids = db_cursor.fetchall()
        all_users_ids = [user_id[0] for user_id in all_users_ids]
        all_users_df = pd.Series(all_users_ids, name="customer_id").to_frame()

        logging.info("Extracting article ids from DB")
        db_cursor.execute("select article_id from articles;")
        all_items_ids = db_cursor.fetchall()
        all_items_ids = [user_id[0] for user_id in all_items_ids]
        all_items_df = pd.Series(all_items_ids, name="article_id").to_frame()

        logging.info("Extracting transactions for model training from DB")
        db_cursor.execute("""
                    select t_dat, customer_id, article_id from transactions_train
                """)
        # where t_dat > '2020-06-21';
        all_transactions = db_cursor.fetchall()
        all_transactions = pd.DataFrame(all_transactions, columns=["t_dat", "customer_id", "article_id"])
        all_transactions["t_dat"] = pd.to_datetime(all_transactions["t_dat"])

        return all_users_df, all_items_df, all_transactions

    def retrain_model(self):

        db_cursor = self.get_db_cursor_()
        all_users_df, all_items_df, all_transactions = self.collect_necessary_data_( db_cursor )
        db_cursor.close()

        logging.info("Initializing ALS recommender model")
        als_recommender = ALSRecommender(all_users_df, all_items_df)
        all_transactions = als_recommender.add_mapped_user_item_ids(all_transactions)

        logging.info("Fitting recommender model...")
        als_recommender.fit_als(all_transactions, **self.als_config)
        logging.info("Done training")

        return als_recommender