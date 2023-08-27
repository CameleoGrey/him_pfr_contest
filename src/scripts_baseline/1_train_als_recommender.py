
import numpy as np
import implicit
from pathlib import Path

import json
import psycopg2 as pg_driver
import clickhouse_driver
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, csr_matrix

from src.classes.utils import *
from src.classes.paths_config import *
from src.classes.ALSRecommender import ALSRecommender

import sys
import logging
logging.basicConfig(level=logging.INFO, filename=Path(log_dir, "model_training.log"), filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

db_drivers = {}
db_drivers["postgre"] = pg_driver
db_drivers["clickhouse"] = clickhouse_driver

logging.info( "Connecting to data base..." )
current_dbms = "postgre"
if current_dbms not in db_drivers:
    logging.error( "Chosen DBMS: {} is not existing. Failed to connect.".format( current_dbms ) )
    raise Exception("Available drivers: {}".format(db_drivers))
with open( Path(configs_dir, "{}_config.json".format(current_dbms)), mode="r" ) as db_config_file:
    db_config = json.load( db_config_file )
db_connect = db_drivers[current_dbms].connect( **db_config )
db_cursor = db_connect.cursor()

logging.info( "Extracting customer ids from DB" )
db_cursor.execute("select customer_id from customers;")
all_users_ids = db_cursor.fetchall()
all_users_ids = [user_id[0] for user_id in all_users_ids]
all_users_df = pd.Series(all_users_ids, name="customer_id").to_frame()

logging.info( "Extracting article ids from DB" )
db_cursor.execute("select article_id from articles;")
all_items_ids = db_cursor.fetchall()
all_items_ids = [user_id[0] for user_id in all_items_ids]
all_items_df = pd.Series(all_items_ids, name="article_id").to_frame()

logging.info( "Extracting transactions for model training from DB" )
db_cursor.execute("""
    select t_dat, customer_id, article_id from transactions_train
""")
#where t_dat > '2020-06-21';
all_transactions = db_cursor.fetchall()
all_transactions = pd.DataFrame( all_transactions, columns=["t_dat", "customer_id", "article_id"] )
all_transactions["t_dat"] = pd.to_datetime( all_transactions["t_dat"] )

logging.info("Initializing ALS recommender model")
als_recommender = ALSRecommender( all_users_df, all_items_df )
all_transactions = als_recommender.add_mapped_user_item_ids( all_transactions )

"""als_recommender.evaluate_als_cv( all_transactions, cv=5, train_days=28, validation_days=7,
                                factors=100, iterations=12,
                                nlist=400, nprobe=20, use_gpu=False,
                                calculate_training_loss=False,
                                regularization=0.01, random_state=45 )"""

logging.info("Fitting recommender model...")
als_recommender.fit_als(all_transactions,
                        factors=500, iterations=3,
                        nlist=400, nprobe=20, use_gpu=False,
                        calculate_training_loss=False,
                        regularization=0.01, random_state=45 )
save( als_recommender, Path(interim_dir, "als_recommender.pkl") )

logging.info("Done training")