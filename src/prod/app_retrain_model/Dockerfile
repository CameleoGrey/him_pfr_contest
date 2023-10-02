# Download image of Ubuntu 22.04
FROM ubuntu:22.04

# To prevent interactive dialogs
ENV TZ=Europe \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y python3.11
RUN apt-get install -y curl

# Install pip for Python 3.11
RUN curl -sS -O https://bootstrap.pypa.io/get-pip.py | python3.11
# THERE'S AN ERROR
RUN python3.11 get-pip.py

WORKDIR app_retrain_model
ADD src/prod/app_retrain_model/main.py main.py
ADD src/prod/app_retrain_model/classes classes
ADD src/prod/app_retrain_model/configs configs
ADD ./requirements.txt requirements.txt

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11000"]

EXPOSE 11000
