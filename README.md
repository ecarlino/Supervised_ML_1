# Supervised_ML_1
Overview of Supervised Machine Learning Algorithms with GridSearchCV, Learning Curves &amp; Model Complexity Curves.

The project consists of 2 parts, each in their own file:

1. Data abstraction. (AbstractData.py)

Since data can be sourced from a variety of places (online, local files, APIs, etc) and since the ML algos simply need data in 4 dataframes (x_train, y_train, x_test, y_test), then it makes sense to abstract the data wrangling necessary to preprocess the source data and expose the 4 dataframes.

One nice feature that has been implmemented is the choice on which method to use to Normalize Continuous Numeric Features.  The 3 options correspond to various Distributions:  (a) Uniform, (b) Gaussian (similar to Standard but with extremes 'clipped') and (c) Standard.


2. ML algos and a standard set of evaluation tools. (SupervisedEval.py)

For each ML algo:  Decision Tree, Boosting, Neural Networks, kNN and SVM -

Start by considering a variety of parameters / parameter values using GridSearchCV and getting some basic stats for the 'best' parameter set.

Then, plot the Learning Curve for each.

Finally, for one or more Hyper Parameters, plot their Model Complexity Curves.
