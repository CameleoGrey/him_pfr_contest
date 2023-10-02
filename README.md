# Littensim-recsys
Littensim (little enterprise simulator) is a series of my projects to train Machine Learning Engineer skills. The big idea is to take an applied problem like recommendations, real dataset and build near enterprise system by using all popular technologies and effective approaches (from baseline to SotA).

# Current research code state:
- [:white_check_mark:] Matrix Factorization baseline **(FaissAlternatingLeastSquares)**
- [:x:] Two-Stage model **(Reranker with boostings)**
- [:x:] Try to use Polars instead Pandas
- [:x:] Try to use **ReckBole**
- [:x:] Classic TwoTower **(NN for recsys)**
- [:x:] TwHIN **(Graph NN approach)**
- [:x:] Online Recommendation technics


# Current production code state:
- [:white_check_mark:] Postgres
- [:white_check_mark:] ClickHouse
- [:x:] GreenPlum
- [:x:] SQLAlchemy
- [:x:] Cassandra
- [:x:] MongoDB
- [:x:] SuperSet
- [:x:] Tableau
- [:white_check_mark:] logging
- [:x:] Tests
- [:white_check_mark:] FastAPI / Flask / DRF
- [:white_check_mark:] asyncio, aiohttp
- [:white_check_mark:] Docker
- [:x:] Docker Compose
- [:white_check_mark:] k3s
- [:x:] AirFlow / Celery
- [:x:] CI/CD (GitLab)
- [:white_check_mark:] Minio
- [:x:] DVC
- [:x:] MLFlow
- [:x:] Redis
- [:x:] Kafka
- [:x:] Hadoop
- [:x:] Spark
- [:x:] HBase
- [:x:] ElasticSearch
- [:x:] LogStash
- [:x:] Kibana
- [:x:] Grafana
- [:x:] Yandex Cloud *(deploy something to train cloud deploy)*
- [:x:] A/B Tests
- [:x:] Scala
- [:x:] Akka

# Preparing for local development
```
git clone https://github.com/CameleoGrey/littensim_recsys.git
conda create -n littensim_recsys python=3.11
conda activate littensim_recsys
pip install -r requirements.txt
```


# Getting Started

- Download [H&M Personal Fashion Recommendation contest data](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
>- For all (prod/research):
>>- Install **PostgeSQL** or **ClickHouse**.
>>- Put the data into your DBMS.
>- For production code
>>- Deploy **Minio** and **Private Registry** on your VMs by **k3s**. *(Look for guides inside docs dir)*
>>- Build app images from src/prod/
>>- Deploy apps from built images
>>- Play with services
