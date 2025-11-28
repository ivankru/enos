Anonymous code for the article

SCENT OF HEALTH: OLFACTORY MULTIVARIAT TIME-SERIES DATASET FOR NON-INVASIVE DISEASE SCREENING

**How to use**
1) Name this folder "enose"

2) For every script except run_exp_article_graphormer_valid.py use python 3.10.16 environment and ```pip install -r req_enose.txt```
For run_exp_article_graphormer_valid.py use python 3.10.18 environment and ```pip install -r req_graphormer.txt```

3) TS2vec benchmark run_exp_article_ts2vec_valid.py should be launched from a ts2vec folder, obtained by git clone https://github.com/zhihanyue/ts2vec.git; ts2vec folder and the folder containing the scripts should both have the same parent directory; copy files from ts2vec_modified_files to the ts2vec folder for the script to run as intended.

4) To run the experiments simply run the script, e.g.:
```python run_exp_article_catboost_valid.py```
It will save the results into a csv file in the working directory.

