# Data Science Nanodegree Project 7: Building a Binary Classifier with Spark

Author: Simon Thornewill von Essen

Data: 2019-07-14

## Description

Predict user churn for a music streaming serivce, "sparkify".

Blog post for this project can be found [here](https://medium.com/@sthornewillve/writing-lighting-fast-code-with-spark-4375a244d128).

## Resulty Summary

Out of the models that were trained and tested, it was found that the best model so far Had roughly 15.8%,	25.4%,	0.70%	and 10.9% accuracy, precision, recall and f1 scores respectively.

Given from these results, we can see that it is difficult to predict when someone will churn then we either need to feed the model more data by sizing up to a full cluster, adding features that provide different patterns of data to what we have already submitted or increase complexity of our model. These are the things I would focus on to improve in the future.

## Repo Layout

The repo layout in this instance is relatively simple. The work done can be found in the jupyter notebook. The json file which contains the data was a little too big to commit and so I have ommitted it.

It should be noted that the folders `./decisionTree1/`, `./logisticRegression1/` and `./logisticRegression2/` are checkpoints for models that are trained in the jupyter notebook. 

Furthermore, `robustness_test.csv` contains a checkpoint for the results of the decision tree model with the optimal hyperparameters found through gridsearch. 

## Packages Used

* `pyspark`
* `time`
* `matplotlib`
* `numpy`
* `pandas`
* `pickle`
* `os`
* `tqdm`

## Acknowledgements

N/A
