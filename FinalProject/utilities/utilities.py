import csv
import numpy as np
import pandas as pd
'''
This function gets the data from CSV file. I prefer using Pandas
'''
def GetData(sample):
        if ( sample == 'train'):
            location = "A:/Spring 2017/AML/project/dataset/adult.data.csv"
        else:
            location = "A:/Spring 2017/AML/project/dataset/adult.test.csv"

    #raw_data = open(location, 'rt', )
        data = pd.read_csv(location, delimiter=',', sep='\n')
        data.replace(' ?', np.nan, inplace=True)
        #data.replace(np.inf, 'NaN')
        #data.fillna('NaN')
        data = data.dropna(axis=0, how="any")
        category_col = data[['workclass', 'marital-status', 'relationship', 'race',  'sex', 'Salary' ]]
        for col in category_col:
            b, c = np.unique(data[col], return_inverse=True)
            data[col] = c

        return data

def GetBreastCancerData():

    location = "A:/Spring 2017/AML/project/dataset/breast-cancer-wisconsin.data.csv"
    data = pd.read_csv(location, delimiter=',', sep='\n')
    data.replace('?', np.nan, inplace=True)
    data = data.dropna(axis=0, how="any")
    data = data.drop('ID', axis=1)
    data['Y'] = data['Y'].map({2:0, 4:1})

    return  data