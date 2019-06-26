# Data Science Nanodegree Project 5: Disaster Response Pipeline Webapp

Author: Simon Thornewill von Essen

Start Date: 2019-06-22

## Description

When natural disasters happen, people use twitter to try and get help. However, not all tweets are natural disaster tweets and there isn't a simple way to use key words in order to identify these kinds of texts.

Towards this end, the task is to create a machine learning webapp that is able to take tweets and discern whether they are relevant for a disaster response team or not using data that has already been labeled.

There are three steps:

1. Create an ETL which cleans the Data
2. Create a ML pipeline which performs feature extraction and trains a model
3. Take model and embed it into a webapp

## Repo Layout

This repo is split into subdirectories:

1. `data-processing_and_model-training` - Contains notebooks, python files and other such files for ML Engineering purposes
2. `webapp` - Contains webapp for deployment, see README inside this directory for more instructions

## Packages Used

* `sys`
* `pandas`
* `sqlalchemy`
* `joblib`
* `re`
* `nltk`
* `sklearn`
* `json`
* `plotly`
* `flask`

## Acknowledgements

* Special thanks to @wenshihao1993, without whom I would have had a lot more trouble getting the python pipeline to work.
