# Hybrid Ranking Network for Text-to-SQL
Code for our paper [Hybrid Ranking Network for Text-to-SQL](https://raw.githubusercontent.com/microsoft/IRNet/master/README.md) 

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