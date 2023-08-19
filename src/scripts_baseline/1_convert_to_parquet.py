

import os
import numpy as np
import pandas as pd

from src_v_1_no_cv.classes.utils import *
from src_v_1_no_cv.classes.paths_config import *
from tqdm import tqdm

files_to_convert = ["articles", "customers", "transactions_train"]

types_dicts = {}
types_dicts["articles"] = {"article_id": str, "product_code": str, "prod_name": str, "product_type_no": str,
                           "product_type_name": str, "product_group_name": str,
                           "graphical_appearance_no": str, "graphical_appearance_name": str,
                           "colour_group_code": str, "colour_group_name": str, "perceived_colour_value_id": str,
                           "perceived_colour_value_name": str, "perceived_colour_master_id": str, "perceived_colour_master_name": str,
                           "department_no": str, "department_name": str, "index_code": str, "index_name": str, "index_group_no": str,
                           "section_no": str, "section_name": str, "garment_group_no": str, "garment_group_name": str,
                           "detail_desc": str}

types_dicts["customers"] = {"customer_id": str, "FN": str,
                            "Active": str, "club_member_status": str,
                            "fashion_news_frequency": str,
                            "age": str, "postal_code": str}

types_dicts["transactions_train"] = {"t_dat": str, "customer_id": str, "article_id": str, "price": float,
                                     "sales_channel_id": str}

for csv_file_name in tqdm(files_to_convert, desc="Converting csv to parquet"):
    readed_df = pd.read_csv( os.path.join( raw_dir, csv_file_name + ".csv" ), dtype=types_dicts[csv_file_name] )
    readed_df.to_parquet( os.path.join( raw_dir, csv_file_name + ".parquet" ) )

print("done")

