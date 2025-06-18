import pandas as pd
from pandas.core import indexing

def load_and_preprocess_data(path: str):
    column_names = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity','median_house_value']

    raw_dataset = pd.read_csv('data/housing.csv',names=column_names,na_values="?",comment='\t',sep=",",index_col=False,skiprows=1)
    dataset = raw_dataset.copy()
    dataset.isna().sum()
    dataset = raw_dataset.dropna()
    
    # One-hot encoding
    
    dataset = pd.get_dummies(dataset, columns=['ocean_proximity'], prefix='ocean_proximity')
    
    # Train-test split
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # Labels
    train_labels = train_dataset.pop('median_house_value')
    test_labels = test_dataset.pop('median_house_value')

    # Normalization stats
    train_stats = train_dataset.describe().transpose()
    
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    
    normed_train = norm(train_dataset)
    normed_test = norm(test_dataset)

    return normed_train, train_labels, normed_test, test_labels, train_stats