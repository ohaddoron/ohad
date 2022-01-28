FROM python:3.9-slim

RUN apt update && apt upgrade -y

COPY src /code
COPY common-code/common /common
COPY config.toml /
COPY requirements-gpu.txt requirements.txt

RUN pip install -U pip

RUN pip install -r requirements.txt


