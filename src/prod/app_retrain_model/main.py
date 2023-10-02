
from pathlib import Path
import json
import logging
import psycopg2 as pg_driver
import clickhouse_driver
from classes.ALSModelRetrainer import ALSModelRetrainer
from classes.utils import *
import os
import sys
import logging
from fastapi import FastAPI
from typing import Union, AnyStr
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from minio import Minio

logging.basicConfig(level=logging.INFO, filename=Path(".", "model_training.log"), filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

db_drivers = {}
db_drivers["postgres"] = pg_driver
db_drivers["clickhouse"] = clickhouse_driver

app = FastAPI()


@app.get("/retrain_model/")
async def retrain_model(current_dbms: str = "postgres",
                        factors: Union[int, None] = None,
                        iterations: Union[int, None] = None,
                        nlist: Union[int, None] = None,
                        nprobe: Union[int, None] = None,
                        regularization: Union[int, None] = None) -> Response:

    logging.info( "Reading configs..." )
    with open(Path("configs", "db_config.json"), mode="r") as db_config_file:
        db_config = json.load( db_config_file )
        db_config = db_config[ current_dbms ]
        logging.info( "Reading db config..." )
    with open(Path("configs", "retrain_config.json"), mode="r") as retrain_config_file:
        retrain_config = json.load( retrain_config_file )
        logging.info( "Reading retrain config..." )
    with open(Path("configs", "services_addresses.json"), mode="r") as services_addresses_file:
        services_addresses_config = json.load( services_addresses_file )
        logging.info( "Reading services addresses config..." )

    logging.info( "Retraining model..." )
    if factors is not None: retrain_config["factors"] = factors
    if iterations is not None: retrain_config["iterations"] = iterations
    if nlist is not None: retrain_config["nlist"] = nlist
    if nprobe is not None: retrain_config["nprobe"] = nprobe
    if regularization is not None: retrain_config["regularization"] = regularization
    logging.info( "Retrain parameters: {}".format( retrain_config ) )
    model_retrainer = ALSModelRetrainer( db_driver = db_drivers[current_dbms],
                                         db_config = db_config,
                                         als_config = retrain_config )
    model = model_retrainer.retrain_model()

    logging.info("Locally serializing model...")
    serialized_model_path = Path(".", "als_recommender.pkl")
    save( model, serialized_model_path )

    logging.info( "Uploading serialized model into object storage..." )
    minio_client = Minio( **services_addresses_config["minio"] )
    bucket_exists = minio_client.bucket_exists( "models" )
    if not bucket_exists:
        logging.info( "Creating bucket \"models\"" )
        minio_client.make_bucket( "models" )
    minio_client.fput_object( bucket_name="models",
                              object_name="als_recommender.pkl",
                              file_path=serialized_model_path )
    logging.info( "Serialized model uploaded." )

    logging.info("Deleting locally serialized model...")
    os.remove( serialized_model_path )
    logging.info("Locally serialized model deleted.")

    response = JSONResponse( content={"message": "Model retrained and updated."} )
    return response