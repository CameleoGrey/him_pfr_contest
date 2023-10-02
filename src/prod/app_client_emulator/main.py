
import asyncio
import collections
import aiohttp
from typing import Union
from fastapi import FastAPI
from pathlib import Path
import random
import psycopg2 as pg_driver
import json
import logging

logging.basicConfig(level=logging.INFO, filename=Path(".", "inference_service.log"), filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")

logging.info("Reading DB config...")
with open( Path(".", "configs", "db_config.json"), "r" ) as db_config_file:
    db_config = json.load( db_config_file )
    db_config = db_config["postgres"]
    logging.info("DB config read.")

logging.info("Extracting user_ids from DB...")
db_connect = pg_driver.connect( **db_config )
db_cursor = db_connect.cursor()
db_cursor.execute( """select customer_id from customers;""" )
user_ids = db_cursor.fetchall()
user_ids = [ user_id[0] for user_id in user_ids ]
db_cursor.close()
logging.info("User_ids extracted.")


app = FastAPI()


@app.get("/batch_requests/")
async def request_recommendations(n_requests: int):

    with open( Path("configs", "services_addresses.json"), "r" ) as addresses_file:
        addresses_config = json.load( addresses_file )
        recsys_inference_address = addresses_config["recsys-inference"]

    requested_url = "http://{}:{}/recommendations/".format( recsys_inference_address["host"],
                                                                  recsys_inference_address["port"])

    async def do_request(session, url, id, i):
        request_path = url + "?user_id={}".format(id) + "&N={}".format( 12 )
        async with session.get( request_path ) as response:
            response_data = await response.text()
            logging.info("Response for i: {} user_id: {}:".format( i, id ))
            logging.info(response_data)

    id_is = []
    for i in range(n_requests):
        id_is.append(i)

    client_session = aiohttp.ClientSession()
    async with client_session as session:
        request_tasks = []
        for i, user_id in zip(id_is, user_ids):
            request_tasks.append( do_request(session, requested_url, user_id, i) )
        await asyncio.gather(*request_tasks)


