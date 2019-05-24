# Data Science Nanodegree Project 4: Write a Data Science Blogpost

Author: Simon Thornewill von Essen

Start Date: 2019-05-23

## Description

While searching for ML projects in the UCI ML repository, I came across a forrest fires dataset which caught my eye. Inside I saw a regression analysis 
using SVMs to predict the total area burned by various fires. Upon looking at the histogram of the target variable, I found that it had a log(x+1) 
distribution similar to another project that Ive seen before, predicing a similar output withan XGBoost model upon a tweedie distribution. I want to try 
and apply this method here and see what happens.

## Repo Layout

This repository will have 3 sub-directories.

* dat
* python
* research
* blog

The dat folder will contain any data used for this analysis, raw or otherwise, stored as CSV files. The python folder will contain a Jupyter notebook of 
the the analysis as well as an HTML version for quick and easy viewing without needing to spin up Jupyter. Finally, when the findings of this analysis 
need to be communicated, a copy of the draft will be found in the blog directory as a markdown file as well as images or any other relevant files.

The research directory stores the paper that did the original fit using SVMs as well as my notes on this research. 

## Packages Used

<To be Updated Before Project Upload>

## Awknowledgements

Data can be found in the [UCI ML repo](https://archive.ics.uci.edu/ml/datasets/Forest+Fires), there is also a 
[Kaggle](https://www.kaggle.com/elikplim/forest-fires-data-set) page for this set.

Citation: [Cortez and Morais, 2007] P. Cortez and A. Morais. A Data Mining Approach to Predict Forest Fires using Meteorological Data. In J. Neves, M. 
F. Santos and J. Machado Eds., New Trends in Artificial Intelligence, Proceedings of the 13th EPIA 2007 - Portuguese Conference on Artificial 
Intelligence, December, Guimarães, Portugal, pp. 512-523, 2007. APPIA, ISBN-13 978-989-95618-0-9.
