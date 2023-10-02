
from pathlib import Path
from fastapi import FastAPI
from typing import Union
import logging
import sys
import json
from minio import Minio
from classes.utils import *

logging.basicConfig(level=logging.INFO, filename=Path(".", "inference_service.log"), filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

with open(Path("configs", "services_addresses.json"), mode="r") as services_addresses_file:
    services_addresses_config = json.load( services_addresses_file )
    logging.info( "Reading services addresses config..." )

logging.info( "Downloading serialized model from object storage..." )
minio_client = Minio( **services_addresses_config["minio"] )
bucket_exists = minio_client.bucket_exists( "models" )
minio_client.fget_object( bucket_name="models",
                          object_name="als_recommender.pkl",
                          file_path=Path(".", "als_recommender.pkl") )
logging.info( "Serialized model downloaded." )

logging.info("Deserializing model")
als_recommender = load( Path(".", "als_recommender.pkl") )
logging.info("Model deserialized")

app = FastAPI()

@app.get("/recommendations/")
async def get_recommendations(user_id: str, N: Union[int, None] = 12):
    response = als_recommender.recommend(user_id, N=N, filter_already_liked_items=False, batch_size=2000)
    return response
