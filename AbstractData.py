from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, ShuffleSplit

from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder

from sklearn import tree, svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import AdaBoostClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class SomeInterestingData:

    def __init__(self, dataSet='badEntry', contNum_Style='badEntry', categ_Style='Categorical', printDataDesc=False):

        self.dataSet = dataSet
        self.contNum_Style = contNum_Style
        self.categ_Style   = categ_Style

        # load correct dataset
        if dataSet == 'Iris':
            self.load_Iris()
        elif dataSet == 'WhiteWine':
            self.load_WhiteWine()
        elif dataSet == 'Wine':
            self.load_Wine()
        elif dataSet == 'BCancer':
            self.load_BCancer()
        elif dataSet == 'BankMktg':
            self.load_BankMktg()
        else:
            raise Exception('Invalid dataSet requested: ', dataSet)

        # if requested, print descriptive stats
        if printDataDesc == True:
            self.print_XY_TestTrain()

    def get_Data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def load_Iris(self):
        # load data via sklearn function
        idata = load_iris()

        # get data & target from sklearn object and convert to pandas dataframes
        data   = pd.DataFrame(idata.data)
        target = pd.DataFrame(idata.target)

        # now split into training / testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, target, test_size = 0.2, random_state = 42, stratify=target)

        # scale Continuous Numerical features
        cnScaler = self.get_ContNum_Scaler(self.x_train)
        self.x_train = cnScaler.transform(self.x_train)
        self.x_test  = cnScaler.transform(self.x_test)

    def load_Wine(self):
        # load from file w/o column headings on first row
        idata = pd.read_csv("S:\\GATECH\\CS-7641-ML\\assignment1\\_data1b_Wine\\wine.dat", sep=",", header=0)
        idata.columns = ["Quality","Alcohol","MalicAcid","Ash","Alcalinity","Magnesium","Phenols","Flavanoids","Nonflavanoid","Proanthocyanins","ColorIntensity","Hue","Diluted","Proline"]

        # now, let's get the data ready for training
        # we need to split the features (X) from the Y
        target = pd.DataFrame(idata.Quality)
        data   = pd.DataFrame(idata.drop('Quality', axis=1))

        # now split into training / testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, target, test_size = 0.2, random_state = 42, stratify=target)

        # scale Continuous Numerical features
        cnScaler = self.get_ContNum_Scaler(self.x_train)
        self.x_train = cnScaler.transform(self.x_train)
        self.x_test  = cnScaler.transform(self.x_test)

        # but still need to compress into (0,1)
        scaler_minmax = preprocessing.MinMaxScaler()
        self.x_train = scaler_minmax.fit_transform(self.x_train)
        self.x_test  = scaler_minmax.transform(self.x_test)

    def load_WhiteWine(self):
        # load from file w/o column headings on first row
        idata = pd.read_csv("S:\\GATECH\\CS-7641-ML\\assignment1\\_data1_Wine\\WhiteQuality.csv", sep=",", header=0)
        idata.columns = ["fixAcidity","volAcidity","citAcid","resSugar","chlorides","freeSulfur","totSulfur","density","pH","sulphates","alcohol","quality"]

        # now, let's get the data ready for training
        # we need to split the features (X) from the Y
        target = pd.DataFrame(idata.quality)
        data   = pd.DataFrame(idata.drop('quality', axis=1))

        # now split into training / testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, target, test_size = 0.2, random_state = 42, stratify=target)

        # scale Continuous Numerical features
        cnScaler = self.get_ContNum_Scaler(self.x_train)
        self.x_train = cnScaler.transform(self.x_train)
        self.x_test  = cnScaler.transform(self.x_test)

        # but still need to compress into (0,1)
        scaler_minmax = preprocessing.MinMaxScaler()
        self.x_train = scaler_minmax.fit_transform(self.x_train)
        self.x_test  = scaler_minmax.transform(self.x_test)

    def load_BCancer(self):
        # load from file w/o column headings on first row
        idata = pd.read_csv("S:\\GATECH\\CS-7641-ML\\assignment1\\_data2_cancer\\cancer_data.csv", sep=",", header=0)
        idata.columns = ['IdNum','Thickness','UniSize','UniShape','Adhesion','EpiSize','BareNuclei','Chromatin','NormNuclei','Mitoses','Diagnosis']

        # IdNum will have no predictive value
        idata = idata.drop('IdNum', axis=1)
        # the BareNuclei column has a few missing data points, denoted by '?' which causes the entire column to be a STRING
        # first, get rid of rows having missing BareNuclei data, then convert the column to numeric
        idata = idata.loc[idata['BareNuclei'] != '?']
        idata.BareNuclei = pd.to_numeric(idata.BareNuclei)

        # features are all numerical and discrete integers in the range 1-10
        # class is Diagnosis:  2 = benign (2/3), 4 = malignant (1/3)
        # changed that to (0,1)
        idata.Diagnosis.replace(2,0, inplace=True)
        idata.Diagnosis.replace(4,1, inplace=True)

        # now, let's get the data ready for training
        # we need to split the features (X) from the Y
        target = pd.DataFrame(idata.Diagnosis)
        data   = pd.DataFrame(idata.drop('Diagnosis', axis=1))

        # now split into training / testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, target, test_size = 0.2, random_state = 42, stratify=target)

        # scale Continuous Numerical features
        cnScaler = self.get_ContNum_Scaler(self.x_train)
        self.x_train = cnScaler.transform(self.x_train)
        self.x_test  = cnScaler.transform(self.x_test)

        # but still need to compress into (0,1)
        scaler_minmax = preprocessing.MinMaxScaler()
        self.x_train = scaler_minmax.fit_transform(self.x_train)
        self.x_test  = scaler_minmax.transform(self.x_test)

    def load_BankMktg(self):
        # load data from disk
        idata = pd.read_csv("S:\\GATECH\\CS-7641-ML\\assignment1\\BankMkting\\bank-full.csv", sep=";", header=0)
        idata.columns = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]

        # now, let's get the data ready for training
        # we need to split the features (X) from the Y
        _target = pd.DataFrame(idata.y)
        data   = pd.DataFrame(idata.drop(['y', 'day', 'month'], axis=1))

        # deal with target data first:  it is in ('yes', 'no') categories
        # convert to categories, which should result in (0,1)
        target = pd.DataFrame()
        target["y"]    = _target.y.astype('category').cat.codes        

        # now split into training / testing sets
        # we're going to rebuild everything later b/c of having all types of features (continuous, categorical, binary)
        # but for now, we need to scale continuous on only the 'training' set
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, target, test_size = 0.2, random_state = 42, stratify=target)

        # scale Continuous Numerical features
        cnCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
        cnScaler = self.get_ContNum_Scaler(self.x_train[cnCols])
        conNum = cnScaler.transform(idata[cnCols])
        conNum = pd.DataFrame(conNum)
        conNum.columns = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']


        # create a new dataframe to hold reformatted categorical features
        categ = pd.DataFrame()
        # work with original dataset
        _categ   = pd.DataFrame(idata, columns=['job', 'marital', 'education', 'contact', 'poutcome'])
        # preprocess binary features, based on Category Style
        if self.categ_Style == 'OneHotEnc':
            categ = pd.concat([categ, pd.get_dummies(_categ.job.astype('category').cat.codes, prefix='job')], axis=1)
            categ = pd.concat([categ, pd.get_dummies(_categ.marital.astype('category').cat.codes, prefix='marital')], axis=1)
            categ = pd.concat([categ, pd.get_dummies(_categ.education.astype('category').cat.codes, prefix='education')], axis=1)
            categ = pd.concat([categ, pd.get_dummies(_categ.contact.astype('category').cat.codes, prefix='contact')], axis=1)
            categ = pd.concat([categ, pd.get_dummies(_categ.poutcome.astype('category').cat.codes, prefix='poutcome')], axis=1)
        else:
            categ["job"]       = _categ.job.astype('category').cat.codes
            categ["marital"]   = _categ.marital.astype('category').cat.codes
            categ["education"] = _categ.education.astype('category').cat.codes
            categ["contact"]   = _categ.contact.astype('category').cat.codes
            categ["poutcome"]  = _categ.poutcome.astype('category').cat.codes


        # create a new dataframe to hold reformatted categorical features
        bincat = pd.DataFrame()
        # work with original dataset
        _bincat  = pd.DataFrame(idata, columns=['default','housing','loan'])    
        # preprocess binary features
        bincat["default_cat"] = _bincat.default.astype('category').cat.codes
        bincat["housing_cat"] = _bincat.housing.astype('category').cat.codes
        bincat["loan_cat"]    = _bincat.loan.astype('category').cat.codes


        # finally, bring everything back together
        data = pd.DataFrame()
        data = pd.concat([conNum, categ, bincat], axis=1)

        # and then, redo the (train/test) SPLIT 
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, target, test_size = 0.2, random_state = 42, stratify=target)


    def get_ContNum_Scaler(self, trainData):

        if self.contNum_Style == 'Uniform':
            _scaler = preprocessing.QuantileTransformer(output_distribution='uniform')
            return _scaler.fit(trainData)
        elif self.contNum_Style == 'Gaussian':
            _scaler = preprocessing.QuantileTransformer(output_distribution='normal')
            return _scaler.fit(trainData)
        elif self.contNum_Style == 'Standard':
            _scaler = preprocessing.StandardScaler()
            return _scaler.fit(trainData)
        else:
            raise Exception('bad contNum_Style passed:', self.contNum_Style, self.dataSet)

    def print_XY_TestTrain(self):

        print("\n\n\nData style:", self.contNum_Style, self.categ_Style)
        print("x_train:")
        print(pd.DataFrame(self.x_train).describe())
        print("x_test:")
        print(pd.DataFrame(self.x_test).describe())
        print("y_train:", type(self.y_train), self.y_train.shape)
        print(self.y_train.values.ravel())
        print("y_test:", type(self.y_test), self.y_test.shape)
        print(self.y_test.values.ravel())
