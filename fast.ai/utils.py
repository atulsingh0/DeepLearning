{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_sample(df, n):\n",
    "    '''\n",
    "        Inp:\n",
    "             df: Pass the dataframe df\n",
    "             n : Fraction of data\n",
    "        Out:\n",
    "             df_sample : dataframe with n% of data df\n",
    "    '''\n",
    "    df_sample = df.sample(frac=n)\n",
    "    return df_sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def dinfo(df):\n",
    "    '''\n",
    "        Inp:\n",
    "            df: Pass the dataframe df\n",
    "        Out:\n",
    "            Std Out below details\n",
    "            print size\n",
    "    '''\n",
    "    \n",
    "    print(\"Size: \",df.size,\"\\n\")\n",
    "    print(\"Columns:\\n\", df.columns,\"\\n\")\n",
    "    print(\"Null Check:\\n\",df.isnull().sum(),\"\\n\")\n",
    "    print(\"Describe:\\n\",df.describe(),\"\\n\")\n",
    "    print(\"Info:\")\n",
    "    print(df.info())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "os.getcwd()\n",
    "df = pd.read_csv(\"..\\..\\datasets\\housing\\housing.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df1 = get_sample(df, 0.1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Size:  206400 \n",
      "\n",
      "Columns:\n",
      " Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',\n",
      "       'total_bedrooms', 'population', 'households', 'median_income',\n",
      "       'median_house_value', 'ocean_proximity'],\n",
      "      dtype='object') \n",
      "\n",
      "Null Check:\n",
      " longitude               0\n",
      "latitude                0\n",
      "housing_median_age      0\n",
      "total_rooms             0\n",
      "total_bedrooms        207\n",
      "population              0\n",
      "households              0\n",
      "median_income           0\n",
      "median_house_value      0\n",
      "ocean_proximity         0\n",
      "dtype: int64 \n",
      "\n",
      "Describe:\n",
      "           longitude      latitude  housing_median_age   total_rooms  \\\n",
      "count  20640.000000  20640.000000        20640.000000  20640.000000   \n",
      "mean    -119.569704     35.631861           28.639486   2635.763081   \n",
      "std        2.003532      2.135952           12.585558   2181.615252   \n",
      "min     -124.350000     32.540000            1.000000      2.000000   \n",
      "25%     -121.800000     33.930000           18.000000   1447.750000   \n",
      "50%     -118.490000     34.260000           29.000000   2127.000000   \n",
      "75%     -118.010000     37.710000           37.000000   3148.000000   \n",
      "max     -114.310000     41.950000           52.000000  39320.000000   \n",
      "\n",
      "       total_bedrooms    population    households  median_income  \\\n",
      "count    20433.000000  20640.000000  20640.000000   20640.000000   \n",
      "mean       537.870553   1425.476744    499.539680       3.870671   \n",
      "std        421.385070   1132.462122    382.329753       1.899822   \n",
      "min          1.000000      3.000000      1.000000       0.499900   \n",
      "25%        296.000000    787.000000    280.000000       2.563400   \n",
      "50%        435.000000   1166.000000    409.000000       3.534800   \n",
      "75%        647.000000   1725.000000    605.000000       4.743250   \n",
      "max       6445.000000  35682.000000   6082.000000      15.000100   \n",
      "\n",
      "       median_house_value  \n",
      "count        20640.000000  \n",
      "mean        206855.816909  \n",
      "std         115395.615874  \n",
      "min          14999.000000  \n",
      "25%         119600.000000  \n",
      "50%         179700.000000  \n",
      "75%         264725.000000  \n",
      "max         500001.000000   \n",
      "\n",
      "Info:\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 20640 entries, 0 to 20639\n",
      "Data columns (total 10 columns):\n",
      "longitude             20640 non-null float64\n",
      "latitude              20640 non-null float64\n",
      "housing_median_age    20640 non-null float64\n",
      "total_rooms           20640 non-null float64\n",
      "total_bedrooms        20433 non-null float64\n",
      "population            20640 non-null float64\n",
      "households            20640 non-null float64\n",
      "median_income         20640 non-null float64\n",
      "median_house_value    20640 non-null float64\n",
      "ocean_proximity       20640 non-null object\n",
      "dtypes: float64(9), object(1)\n",
      "memory usage: 1.6+ MB\n",
      "None \n",
      "\n"
     ]
    }
   ],
   "source": [
    "dinfo(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 20640 entries, 0 to 20639\n",
      "Data columns (total 10 columns):\n",
      "longitude             20640 non-null float64\n",
      "latitude              20640 non-null float64\n",
      "housing_median_age    20640 non-null float64\n",
      "total_rooms           20640 non-null float64\n",
      "total_bedrooms        20433 non-null float64\n",
      "population            20640 non-null float64\n",
      "households            20640 non-null float64\n",
      "median_income         20640 non-null float64\n",
      "median_house_value    20640 non-null float64\n",
      "ocean_proximity       20640 non-null object\n",
      "dtypes: float64(9), object(1)\n",
      "memory usage: 1.6+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " Volume in drive C is OS\n",
      " Volume Serial Number is D202-8009\n",
      "\n",
      " Directory of C:\\learn\\Git\\DeepLearning\\fast.ai\n",
      "\n",
      "10-01-2018  16:39    <DIR>          .\n",
      "10-01-2018  16:39    <DIR>          ..\n",
      "10-01-2018  16:14    <DIR>          .ipynb_checkpoints\n",
      "10-01-2018  16:13             5,400 utils.ipynb\n",
      "10-01-2018  16:39             7,461 utils.py\n",
      "               2 File(s)         12,861 bytes\n",
      "               3 Dir(s)  226,279,440,384 bytes free\n"
     ]
    }
   ],
   "source": [
    "!dir"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
