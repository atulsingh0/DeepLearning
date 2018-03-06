# import
import numpy as np
import pandas as pd
import os
import glob
import shutil
import stat
import pathlib
import pickle
import matplotlib.pyplot as plt



def df_sample(df, n):
    '''
    Fetch sample data from dataframe
        df: Pass the dataframe df
        n : Fraction of data
        df_sample : dataframe with n% of data df
    '''
    df_sample = df.sample(frac=n)
    return df_sample


def df_summary(df):
    '''
    Return dataframe info
    '''
    print("Summary :")
    return (df.info())


def df_null(df):
    '''Null Check in DataFrame'''
    return df.isnull().sum()


def df_null_per(df):
    '''Null Percentage Check in DataFrame'''
    return df.isnull().sum().sort_index()/len(df)


def df_remNull(df, p, type=''):
    '''
    Remove the Columns which is having more than
    p% NUll in it

    df: Input DataFrame
    p : Percentage, i.e. -  0.1 = 10%
    type : 'r' for row, 'c' for column
            By default it will remove columns as well as row
    '''
    #return df[df.columns[list(df.isnull().sum()/df.size < p )]]
    if type=='c':
         df2 = df.loc[:, df.isnull().sum()/len(df) < p ]
    elif type=='r':
        df2 = df.loc[(df.isnull().transpose().sum()/len(df) < p ).index]
    else:
        df2 = df.loc[:, df.isnull().sum()/len(df) < p ]
        df2 = df2.loc[:, df.isnull().sum()/len(df) < p ]
    return df2


def df_remColNull(df, p):
    '''
    Remove the Columns which is having more than
    p% NUll in it

    df: Input DataFrame
    p : Percentage, , 0.1 = 10%
    '''
    #return df[df.columns[list(df.isnull().sum()/df.size < p )]]
    return df.loc[:, df.isnull().sum()/len(df) < p ]


def df_remRowNull(df, p):
    '''
    Remove the Row which is having more than
    p% NUll in it

    df: Input DataFrame
    p : Percentage, 0.1 = 10%
    '''
    return df.loc[(df.isnull().transpose().sum()/len(df) < p ).index]


def _remove_readonly(fn, path_, excinfo):
    # Handle read-only files and directories
    if fn is os.rmdir:
        os.chmod(path_, stat.S_IWRITE)
        os.rmdir(path_)
    elif fn is os.remove:
        os.lchmod(path_, stat.S_IWRITE)
        os.remove(path_)


def force_remove_file_or_symlink(path_):
    try:
        os.remove(path_)
    except OSError:
        os.lchmod(path_, stat.S_IWRITE)
        os.remove(path_)


# Code from shutil.rmtree()
def is_regular_dir(path_):
    try:
        mode = os.lstat(path_).st_mode
    except os.error:
        mode = 0
    return stat.S_ISDIR(mode)


def clear_dir(path_):
    if is_regular_dir(path_):
        # Given path is a directory, clear its content
        for name in os.listdir(path_):
            fullpath = os.path.join(path_, name)
            if is_regular_dir(fullpath):
                shutil.rmtree(fullpath, onerror=_remove_readonly)
            else:
                force_remove_file_or_symlink(fullpath)
    else:
        # Given path is a file or a symlink.
        # Raise an exception here to avoid accidentally clearing the content
        # of a symbolic linked directory.
        raise OSError("Cannot call clear_dir() on a symbolic link")


def make_dirs(dir):
    """
    dir : path of directory
    Do not throw error if dir exists
    """
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)


def dataCategorizer(catg, path):
    """
    Categorizing the data files in different folders
    based on the categories list

    i.e.-
    path is contains 2 kind of files - dogs n cats
    catg = ['dog','cat']
    path = 'path'

    create 2 folders in $path - dog and cat
    and move all the respective file in it
    """
    os.chdir(path)
    [make_dirs(dir) for dir in catg]
    for name in catg:
        for f in glob.glob(name+"*"):
              shutil.move(f, name)


def data_sampler(n, src, tgt):
    """
    Random Sample of files....
    n : No of files needed as sample
    src : Source directory
    tgt : Target directory
    """

    # creating tgt directory if not existing
    make_dirs(tgt)
    files=np.random.choice(os.listdir(src), n, replace=False)
    #print(files)
    for file in files:
        shutil.move(os.path.join(src,file), tgt)
    print("Total Files moved to ",tgt," : ", len(files))


def data_splitter(src, tgt, nsample, nvalid, labels=None):
    """
    Dividing the data into Train, Valid & Sample
    """

    # creating directories
    print("Creating train_1, valid_1 and sample_1 directories....")
    train_dir=os.path.join(tgt,'train_1')
    valid_dir=os.path.join(tgt,'valid_1')
    sample_dir=os.path.join(tgt,'sample_1')

    make_dirs(train_dir)
    make_dirs(valid_dir)
    make_dirs(sample_dir)

    # copying all files into train_1
    print("Copying all the files into train_1....")
    for filename in glob.glob(os.path.join(src, '*.*')):
        shutil.copy(filename, train_dir)
    print("Total Files are: ", len(os.listdir(train_dir)))


    # now categorizing
    if labels:
        print("Categorizing the data as per labels")
        dataCategorizer(labels, train_dir)

    # Moving Files
    print("Moving files now....")
    if labels:
        for label in labels:
            data_sampler(nvalid, os.path.join(train_dir, label), os.path.join(valid_dir, label))
            data_sampler(nsample, os.path.join(train_dir, label), os.path.join(sample_dir, label))
    else:
        data_sampler(nvalid, train_dir, valid_dir)
        data_sampler(nsample, train_dir, sample_dir)


def df_save(df, path, type='c'):
    '''
    To Save any dataset as pickle
    df   : Any data type variable
    path : Full Path with filename i.e. - /tmp/data.raw
    type : default as c
           Values -  c for csv or p for pickle
    '''
    if type=='c':pd.to_csv(path)
    if type=='p': pickle.dump(df, open(path, 'wb'))
    return None


def df_read(path, type='c'):
    '''
    path : Full Path with filename i.e. - /tmp/data.raw
    type : default as c
           VValues -  c for csv or p for pickle
    '''
    if type=='c':data = pd.read_csv(path)
    if type=='p': data = pickle.load(open(path, 'rb'))
    return data


def rmse(x,y):
    '''
    Calculate Root mean Square Error for x and y

    x,y : Input Values
    '''
    return np.sqrt(np.mean(np.square(x-y)))


def split_vals(X,y,p):
    '''
    Split the data into train and test sets by p%

    X : Input X
    y : Input y
    p : Percentage, i.e- 10% - 0.1
    '''
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


def plot_keras_acc_ax(history):
    history = history.history
    train_acc = history['acc']
    val_acc = history['val_acc']
    train_loss = history['loss']
    val_loss = history['val_loss']

    epochs = np.arange(1, len(train_acc)+1)
    hr = np.arange(-1, len(train_acc)+2)
    train_loss_avg = np.repeat(np.mean(train_loss), len(hr))
    val_loss_avg = np.repeat(np.mean(val_loss), len(hr))
    train_acc_avg = np.repeat(np.mean(train_acc), len(hr))
    val_acc_avg = np.repeat(np.mean(val_acc), len(hr))

    fig, ax = plt.subplots(figsize=(22,9), ncols=2)

    ax[0].plot(epochs, train_loss, '.-', label='Train Loss')
    ax[0].plot(epochs, val_loss, '-', label='Validation Loss')
    ax[0].plot(hr, train_loss_avg, '--', label='Train Loss Mean')
    ax[0].plot(hr, val_loss_avg, '--', label='Validation Loss Mean')
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("Loss")
    ax[0].grid()
    ax[0].legend()
    #plt.show()

    #plt.clf()

    ax[1].plot(epochs, train_acc, '.-', label='Train Accuracy')
    ax[1].plot(epochs, val_acc, '-', label='Validation Accuracy')
    ax[1].plot(hr, train_acc_avg, '--', label='Train Loss Mean')
    ax[1].plot(hr, val_acc_avg, '--', label='Validation Loss Mean')
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid()
    ax[1].legend()
    plt.show()


def plot_keras_acc(history):
    history = history.history
    train_acc = history['acc']
    val_acc = history['val_acc']
    train_loss = history['loss']
    val_loss = history['val_loss']

    epochs = np.arange(1, len(train_acc)+1)
    hr = np.arange(-1, len(train_acc)+2)
    train_loss_avg = np.repeat(np.mean(train_loss), len(hr))
    val_loss_avg = np.repeat(np.mean(val_loss), len(hr))
    train_acc_avg = np.repeat(np.mean(train_acc), len(hr))
    val_acc_avg = np.repeat(np.mean(val_acc), len(hr))

    plt.plot(epochs, train_acc, '.-', label='Train Accuracy')
    plt.plot(epochs, val_acc, '-', label='Validation Accuracy')
    plt.plot(hr, train_acc_avg, '.-', label='Train Accuracy Loss')
    plt.plot(hr, val_acc_avg, '-', label='Validation Accuracy Loss')
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.show()

    plt.clf()

    plt.plot(epochs, train_loss, '.-', label='Train Loss')
    plt.plot(epochs, val_loss, '-', label='Validation Loss')
    plt.plot(hr, train_loss_avg, '--', label='Train Loss Mean')
    plt.plot(hr, val_loss_avg, '--', label='Validation Loss Mean')

    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()


def moving_average(a, n=3) :
    '''
    Calculate the moving average for ocuurance n

    a : list like Values
    n : no of occurance
    '''
    a = np.concatenate((np.zeros(n-1), a))
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def moving_average_exp(a, n=3):
    '''
    Calculate the exponential moving average for ocuurance n

    a : list like Values
    n : no of occurance
    '''
    frac = (1 - (1/float(n)))

    lis = np.zeros((a.shape[0] + 1))
    for idx, val in enumerate(a):
        lis[idx] = frac*lis[idx-1] + (1-frac)*val
    ret = lis[:-1]
    return ret
