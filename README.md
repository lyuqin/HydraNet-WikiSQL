# Hybrid Ranking Network for Text-to-SQL
Code for our paper [Hybrid Ranking Network for Text-to-SQL](https://www.microsoft.com/en-us/research/publication/hybrid-ranking-network-for-text-to-sql/) 

## Environment Setup

* `Python3.7`
* `Pytorch 1.3.1` or higher

Once Python and Pytorch is installed, run `pip install -r requirements.txt` to install dependent Python packages.


## Training
1. Download WikiSQL data to `data/wikisql` folder, and run `python wikisql_gendata.py` to preprocess data.
2. Run `python main.py train --conf conf/wikisql.conf --gpu 0,1,2,3 --note "some note"`.
3. Model will be saved to `output` folder, named by training start datetime.

## Evaluate
1. Modify model, input and output settings in `wikisql_prediction.py` and run it.
2. Run WikiSQL evaluation script to get official numbers.

## Trained Model
Trained model that can generate reported number on WikiSQL leaderboard is available [here](https://drive.google.com/file/d/1scefU7X0X-m3-sU3mW-HGyc8ngYNZHmn/view?usp=sharing). 