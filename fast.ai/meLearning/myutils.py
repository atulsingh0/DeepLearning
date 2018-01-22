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