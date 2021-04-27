FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

RUN mkdir data
RUN apt-get update && apt-get install -y wget && apt-get install -y git
RUN git clone https://github.com/salesforce/WikiSQL && tar xvjf WikiSQL/data.tar.bz2 -C WikiSQL
RUN python wikisql_gendata.py
