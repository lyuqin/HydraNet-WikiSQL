# Hybrid Ranking Network for Text-to-SQL
Code for our paper [Hybrid Ranking Network for Text-to-SQL](https://arxiv.org/abs/2008.04759) 

## Environment Setup

* `Python 3.8`
* `Pytorch 1.7.1` or higher
* `pip install -r requirements.txt`

We can also run experiments with docker image:
`docker build -t hydranet -f Dockerfile .`

The built image above contains processed data and is ready for training and evaluation.

## Data Preprocessing
1. Create data folder and output folder first: `mkdir data && mkdir output`
2. Clone WikiSQL repo: 
`git clone https://github.com/salesforce/WikiSQL && tar xvjf WikiSQL/data.tar.bz2 -C WikiSQL`
3. Preprocess data:
`python wikisql_gendata.py`
   
## Training
1. Run `python main.py train --conf conf/wikisql.conf --gpu 0,1,2,3 --note "some note"`.
2. Model will be saved to `output` folder, named by training start datetime.

## Evaluation
1. Modify model, input and output settings in `wikisql_prediction.py` and run it.
2. Run WikiSQL evaluation script to get official numbers: `cd WikiSQL && python evaluate.py data/test.jsonl data/test.db ../output/test_out.jsonl`

Note: the WikiSQL evaluation script will encounter error when running in Windows system. Hence we included the fixed version for Windows User (run in root folder): `python wikisql_evaluate.py WikiSQL/data/test.jsonl WikiSQL/data/test.db output/test_out.jsonl`


## Trained Model
Trained model that can reproduce reported number on WikiSQL leaderboard is attached in the releases (see under "Releases" in the right column). Model prediction outputs are also attached.