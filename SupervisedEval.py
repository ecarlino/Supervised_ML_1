from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, validation_curve

from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder

from sklearn import tree, svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import AdaBoostClassifier

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from InterestingData import SomeInterestingData


def gen_LearnCurve(algo, title, X, y, train_sizes=np.linspace(.1, 1.0, 5)):

    train_sizes, train_scores, test_scores = learning_curve(algo, X, y, cv=5, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    # plot +/- Stdev bands
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plot 'mean'
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def gen_GridSearch(ml_algo, hParams, scoringMethod, x_train, x_test, y_train, y_test):

    # setup & run Grid Search
    gsModel = GridSearchCV(ml_algo, hParams, cv=5,scoring=scoringMethod)
    gsModel.fit(x_train, y_train.values.ravel())

    #results
    print("\nScores by Hparameter set:", scoringMethod)
    means = gsModel.cv_results_['mean_test_score']
    stds = gsModel.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gsModel.cv_results_['params']):
        print("%0.4f (+/-%0.04f) for %r" % (mean, std * 2, params))

    print("\nClassification report:")
    y_true, y_pred = y_test.values.ravel(), gsModel.predict(x_test)
    print(classification_report(y_true, y_pred))

    print("\nBest Hparameter set:", gsModel.best_params_)
    print("Accuracy:",accuracy_score(y_pred,y_true))
    print("Confusion Matrix:\n",confusion_matrix(y_pred,y_true))


def setup_GridSearch(ml_algo):

    if ml_algo == 'DTree':
        algo = tree.DecisionTreeClassifier()
        '''
        max_depth(int):                 NEED TO FIGURE OUT HOW DEEP THESE THINGS ARE GOING FIRST!!!  The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        min_samples_split(int, def:2):  The minimum number of samples required to split an internal node
        min_samples_leaf(int, def:1):   The minimum number of samples required to be at a leaf node
        max_features(auto, sqrt, log2): The number of features to consider when looking for the best split:
        random_state:                   Set for reproducability
        '''
        params = {'min_samples_split': [2,3,5,8,13], 
                  'min_samples_leaf':[1,2,3,5,8,13],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'random_state':[429]}
    elif ml_algo == 'NN':        
        algo = MLPClassifier()
        '''
        hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)  The ith element represents the number of neurons in the ith hidden layer.
        activation    : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
        solver        : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’   Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, ‘lbfgs’ can converge faster and perform better.
        max_iter      : int, optional, default 200   Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations.
        random_state  : Set for reproducability
        '''
        params = {'hidden_layer_sizes': [(25,50,25), (50,50,50), (50,100,50), (50,), (100,)],
                  'activation': ['logistic', 'tanh', 'relu'],
                  'solver': ['lbfgs', 'adam'],
                  'max_iter': [100,200,400,800],
                  'random_state':[429]}
    elif ml_algo == 'Boosting':        
        algo = AdaBoostClassifier()
        '''
        base_estimator - use the default 'DecisionTreeClassifier'
        n_estimators : integer, optional (default=50)   The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
        random_state  : Set for reproducability
        '''
        params = {'n_estimators': [10, 25, 50, 100],
                  'random_state':[429]}
    elif ml_algo == 'SVM':        
        algo = svm.SVC()
        '''
        C : float, optional (default=1.0)   Penalty parameter C of the error term
        kernel : string, optional (default=’rbf’)   Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.        
        max_iter : int, optional (default=-1)       -- need to figure out how many this thing is doing first!!
        random_state  : Set for reproducability
        '''
        params = {'C': [1, 10, 100, 1000], 
                  'kernel': ['rbf','linear'],
                  'random_state':[429]}
    elif ml_algo == 'KNN':        
        algo = KNeighborsClassifier()
        '''
        n_neighbors : int, optional (default = 5)
        weights : str or callable, optional (default = ‘uniform’)        ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally. ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors 
        n_jobs : int, optional (default = 1)        The number of parallel jobs to run for neighbors search. If -1, then the number of jobs is set to the number of CPU cores
        '''
        params = {'n_neighbors':[5,6,7,9,11,14], 
                  'weights':['uniform', 'distance'],
                  'n_jobs':[-1]}
    else:
        raise Exception('invalid algo passed to setup_GridSearch()', ml_algo)

    return algo, params

def lCurveParams(ml_algo, dataSet='Other'):

    # well, this doesn't even work - complete waste of time!
    # need to figure out how to add parameters to function via a variable!

    if dataSet == 'BCancer':
        if ml_algo == 'DTree':
            algoParams = "tree.DecisionTreeClassifier(max_features=auto, min_samples_leaf=3, min_samples_split=2, random_state=429)"
        elif ml_algo == 'NN':        
            algoParams = "MLPClassifier(activation=tanh, hidden_layer_sizes=(50,), max_iter=100, random_state=429, solver=adam)"
        elif ml_algo == 'Boosting':        
            algoParams = "AdaBoostClassifier(n_estimators=10, random_state=429)"
        elif ml_algo == 'SVM':        
            algoParams = "svm.SVC(C=1, kernel=rbf, random_state=429)"
        elif ml_algo == 'KNN':        
            algoParams = "KNeighborsClassifier(n_jobs=-1, n_neighbors=7, weights=uniform)"
        else:
            raise Exception('invalid algo passed to lCurveParams()', ml_algo)
    else:
        if ml_algo == 'DTree':
            algoParams = "tree.DecisionTreeClassifier()"
        elif ml_algo == 'NN':        
            algoParams = "MLPClassifier()"
        elif ml_algo == 'Boosting':        
            algoParams = "AdaBoostClassifier()"
        elif ml_algo == 'SVM':        
            algoParams = "svm.SVC()"
        elif ml_algo == 'KNN':        
            algoParams = "KNeighborsClassifier()"
        else:
            raise Exception('invalid algo passed to lCurveParams()', ml_algo)

    return algoParams


def run_InitTests(interestingData):
    # get 4 data sets
    x_train, x_test, y_train, y_test = interestingData.get_Data()

    nTargetLabels = len(np.unique(y_train))
    if nTargetLabels == 2:
        scoreMethod = 'f1'          # for binary categories
    else:
        scoreMethod = 'f1_micro'    # for more than 2 labels

    #DTree
    ml_algo, hParameters = setup_GridSearch('DTree')
    gen_GridSearch(ml_algo, hParameters, scoreMethod, x_train, x_test, y_train, y_test)
    # now, run Learning Curve with those 'best' parameters  *at this time, just running w/ default params!!
    gen_LearnCurve(ml_algo, "Learning Curve for DTree", x_train, y_train.values.ravel())

    #NN
    ml_algo, hParameters = setup_GridSearch('NN')
    gen_GridSearch(ml_algo, hParameters, scoreMethod, x_train, x_test, y_train, y_test)
    # now, run Learning Curve with those 'best' parameters
    gen_LearnCurve(ml_algo, "Learning Curve for NN", x_train, y_train.values.ravel())

    #Boosting
    ml_algo, hParameters = setup_GridSearch('Boosting')
    gen_GridSearch(ml_algo, hParameters, scoreMethod, x_train, x_test, y_train, y_test)
    # now, run Learning Curve with those 'best' parameters
    gen_LearnCurve(ml_algo, "Learning Curve for Boosting", x_train, y_train.values.ravel())

    # SVM
    ml_algo, hParameters = setup_GridSearch('SVM')
    gen_GridSearch(ml_algo, hParameters, scoreMethod, x_train, x_test, y_train, y_test)
    # now, run Learning Curve with those 'best' parameters
    gen_LearnCurve(ml_algo, "Learning Curve for SVM", x_train, y_train.values.ravel())

    #KNN
    ml_algo, hParameters = setup_GridSearch('KNN')
    gen_GridSearch(ml_algo, hParameters, scoreMethod, x_train, x_test, y_train, y_test)
    # now, run Learning Curve with those 'best' parameters
    gen_LearnCurve(ml_algo, "Learning Curve for KNN", x_train, y_train.values.ravel())

def get_ValidationCurve(algo, hparam, hpValues, X, y, title):

    train_scores, test_scores = validation_curve(algo, X, y, param_name=hparam, param_range=hpValues,cv=10, scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel("Hyperparameter")
    plt.ylabel("Score")

    # plot +/- Stdev bands
    plt.fill_between(hpValues, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(hpValues, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # plot 'mean'
    plt.plot(hpValues, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(hpValues, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

def gen_ValidationCurves(interestingData):

    # get the data for this set
    x_train, x_test, y_train, y_test = interestingData.get_Data()
    
    #Decision Tree
    algo = tree.DecisionTreeClassifier()
    hparam = "max_depth"
    hpVals = [2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50]
    get_ValidationCurve(algo, hparam, hpVals, x_train, y_train.values.ravel(), "Validation Curve for DTree")
    
    #NN
    algo = MLPClassifier()
    hparam = "max_iter"
    hpVals = [10, 25, 50, 100, 200, 300, 400, 600, 800]
    get_ValidationCurve(algo, hparam, hpVals, x_train, y_train.values.ravel(), "Validation Curve for NN")

    #Boosting
    algo = AdaBoostClassifier()
    hparam = "n_estimators"
    hpVals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    get_ValidationCurve(algo, hparam, hpVals, x_train, y_train.values.ravel(), "Validation Curve for Boosting")

    #SVM
    algo = svm.SVC()
    hparam = "C"
    hpVals = [5, 10, 25, 50, 100, 250]
    get_ValidationCurve(algo, hparam, hpVals, x_train, y_train.values.ravel(), "Validation Curve for SVM")

    #KNN
    algo = KNeighborsClassifier()
    hparam = "n_neighbors"
    hpVals = [1,2,3,4,6,8,10,13,16,20, 25, 30]
    get_ValidationCurve(algo, hparam, hpVals, x_train, y_train.values.ravel(), "Validation Curve for KNN")


def main():

    # run all for BCancer - Uniform
    anInterestingDS = SomeInterestingData('BCancer', contNum_Style='Uniform', printDataDesc=True)
    run_InitTests(anInterestingDS)
    gen_ValidationCurves(anInterestingDS)

    # run all for Wine - Uniform
    anInterestingDS = SomeInterestingData('Wine', contNum_Style='Uniform', printDataDesc=True)
    run_InitTests(anInterestingDS)
    gen_ValidationCurves(anInterestingDS)

    # run all for BCancer - Standard
    anInterestingDS = SomeInterestingData('BCancer', contNum_Style='Standard', printDataDesc=True)
    run_InitTests(anInterestingDS)
    gen_ValidationCurves(anInterestingDS)

    # run all for Wine - Standard
    anInterestingDS = SomeInterestingData('Wine', contNum_Style='Standard', printDataDesc=True)
    run_InitTests(anInterestingDS)
    gen_ValidationCurves(anInterestingDS)

    '''
    # run all for White Wine - 
    anInterestingDS = SomeInterestingData('WhiteWine', contNum_Style='Standard', printDataDesc=True)
    run_InitTests(anInterestingDS)
    gen_ValidationCurves(anInterestingDS)


    # run all for BankMktg - 
    anInterestingDS = SomeInterestingData('BankMktg', contNum_Style='Standard', categ_Style='OneHotEnc', printDataDesc=True)
    run_InitTests(anInterestingDS)


    # run all for IRIS - 
    anInterestingDS = SomeInterestingData('Iris', contNum_Style='Standard', printDataDesc=True)
    run_InitTests(anInterestingDS)
    '''

if __name__ == '__main__':
    main()

