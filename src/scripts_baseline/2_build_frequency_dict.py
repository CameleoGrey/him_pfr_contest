
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from src_v_1_no_cv.classes.paths_config import *
from src_v_1_no_cv.classes.utils import *

all_users_df = pd.read_parquet( Path( raw_dir, "customers.parquet" ), columns=["customer_id"] )
all_transactions = pd.read_parquet( Path( raw_dir, "transactions_train.parquet" ),
                                    columns=["t_dat", "customer_id", "article_id"] )

"""all_user_ids = set(list(all_users_df["customer_id"].to_numpy()))
buyers_only = set(list(all_transactions["customer_id"].to_numpy()))
no_buyers = all_user_ids.difference( buyers_only )
no_buyers = list(sorted(list( no_buyers )))
save( no_buyers, Path(interim_dir, "no_buyers_ids.pkl") )"""

customer_ids = all_transactions["customer_id"].to_numpy()
article_ids = all_transactions["article_id"].to_numpy()
frequency_dict = {}
for i in tqdm(range(customer_ids.shape[0]), desc="Building frequency dict"):
    current_customer_id = customer_ids[i]
    current_article_id = article_ids[i]

    if current_customer_id not in frequency_dict:
        frequency_dict[current_customer_id] = {}
    if current_article_id not in frequency_dict[current_customer_id]:
        frequency_dict[current_customer_id][current_article_id] = 0

    frequency_dict[current_customer_id][current_article_id] += 1
save( frequency_dict, Path(interim_dir, "frequency_dict.pkl") )

print("done")