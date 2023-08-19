
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src_v_2_pandas.classes.utils import *
from src_v_2_pandas.classes.paths_config import *

als_recommendations = load( Path( interim_dir, "als_recommendations.pkl" ) )

user_ids = []
predicts = []
for user_id in tqdm(als_recommendations.keys(), desc="Building predicts"):
    user_ids.append( user_id )
    user_items_predict = " ".join( als_recommendations[user_id] )
    predicts.append( user_items_predict )

print("Building submission dataframe")
submission_df = pd.DataFrame( data=[user_ids, predicts] )
submission_df = submission_df.transpose()
submission_df.columns = ["customer_id", "prediction"]

print("Saving to csv")
submission_df.to_csv( Path(result_dir, "submission.csv"), index=False )

print("done")