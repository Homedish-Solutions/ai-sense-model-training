import pandas as pd

def load_and_preprocess_data(path: str):
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(path, names=column_names,
                              na_values = "?", comment='\t',
                              sep=",", index_col=False, skiprows=1)
    
    dataset = raw_dataset.dropna()
    
    # One-hot encoding
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0
    
    # Train-test split
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # Labels
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    # Normalization stats
    train_stats = train_dataset.describe().transpose()
    
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    
    normed_train = norm(train_dataset)
    normed_test = norm(test_dataset)

    return normed_train, train_labels, normed_test, test_labels, train_stats
