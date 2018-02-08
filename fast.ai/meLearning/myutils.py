# import
import numpy as np
import pandas as pd
import os
import glob
import shutil


def get_sample(df, n):
    '''
        df: Inp - Pass the dataframe df
        n : Inp - Fraction of data
        df_sample : Out - dataframe with n% of data df
    '''
    df_sample = df.sample(frac=n)
    return df_sample


def df_summary(df):
    print("Summary :")
    return (df.info())

    
def df_null(df):
    """Null Check in DataFrame"""
    return df.isnull().sum()


def df_null_per(df):
    """Null Percentage Check in DataFrame"""
    return df.isnull().sum().sort_index()/len(df)


def rem_col_null(df, p):
    '''
    Remove the Columns which is having more than
    p% NUll in it
    
    df: Inp - Input DataFrame
    p : Inp - Percentage, , 0.1 = 10%
    '''   
    #return df[df.columns[list(df.isnull().sum()/df.size < p )]]
    return df.loc[:, df.isnull().sum()/len(df) < p ]


def rem_row_null(df, p):
    '''
    Remove the Row which is having more than
    p% NUll in it
    
    df: Inp - Input DataFrame
    p : Inp - Percentage, 0.1 = 10%
    '''
    return df.loc[(df.isnull().transpose().sum()/len(df) < p ).index]
	
	
def dataCategorizer(lst, path):
    os.chdir(path)
    [os.makedirs(dir) for dir in lst]
    for name in lst:
        for f in glob.glob(name+"*"):
              shutil.move(f, name)


def to_save(df, path):
    """To Save any dataset
    df   : Inp - Any data type variable
    path : Inp - Full Path with filename i.e. - /tmp/data.raw
    """
    pickle.save(df, open(path, 'wb'))
    return None
    

def to_read(path):
    """
    path : Inp - Full Path with filename i.e. - /tmp/data.raw
    """
    return pickle.load(open(path, 'rb'))


def rmse(x,y):
    """
    x,y : Inp - Input Values
    """
    return np.sqrt(np.mean(np.square(x-y)))
    
    
def split_vals(X,y,p):
    """
    X : Inp - Input X
    y : Inp - Input y
    p : Inp - Percentage, i.e- 10% - 0.1
    """
    n = len(X) - int(np.ceil(len(X)*p))
    X_train, X_test = X[:n].copy(), X[n:].copy()
    y_train, y_test = y[:n].copy(), y[n:].copy()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test
	

def print_score(clf, X_train, y_train, X_test, y_test):
	"""
	clf : Inp - Model
	X_test, X_test, y_train, y_test : Inp - Data Variable
	res : Out - scores
	"""
	res = [rmse(clf.predict(X_train), y_train), rmse(clf.predict(X_test), y_test), clf.score(X_train, y_train), clf.score(X_test, y_test)]
	if hasattr(clf, 'oob_score_'):
		res = res.append(clf.oob_score_)
		
	return res
	

def reverse_dict(dic):
	'''
	return reverse dictionary
	'''
	rev = dict([(value, key) for (key,value) in dic.items()])
	return rev
	
	

		

