from fancyimpute import KNN
import csv
import numpy as np
import h5py


def read_csv_file(file_path):
    data = []
    f = open(file_path)
    for i, row in enumerate(csv.reader(f)):
        if i != 0:
            data.append(row)
    return np.array(data)


train_path = '/home/cougarnet.uh.edu/amobiny/Desktop/zillow_project/Data Science ZExercise_TRAINING_CONFIDENTIAL1.csv'
test_path = '/home/cougarnet.uh.edu/amobiny/Desktop/zillow_project/Data Science ZExercise_TEST_CONFIDENTIAL2.csv'

train_data = read_csv_file(train_path)
test_data = read_csv_file(test_path)

# replace all the missing values in the 13th column with 0
train_data[train_data[:, 13] == 'NA', 13] = 0
test_data[test_data[:, 13] == 'NA', 13] = 0


# get month from the dates
# def get_month(x):
#     return x.split('/')[0]
#
#
# train_data[:, 2] = np.array(list(map(get_month, train_data[:, 2])))
# test_data[:, 2] = np.array(list(map(get_month, test_data[:, 2])))

# take the training labels
y_train = train_data[:, 1].astype(np.float32)

# delete some useless columns
X_train = np.delete(train_data, [0, 1, 2, 3, 4, 5], axis=1)
X_test = np.delete(test_data, [0, 1, 2, 3, 4, 5], axis=1)

# replace 'NA' with NaN
X_train[X_train == 'NA'] = np.nan
X_test[X_test == 'NA'] = np.nan

# convert values to float
X_train = X_train.astype(np.float32)
X_train = X_train.astype(np.float32)

# Use 5 nearest rows which have a feature to fill in each row's missing features
data_incomplete = np.concatenate((X_train, X_test), axis=0)
data_complete = KNN(k=5).fit_transform(data_incomplete)
X_train = data_complete[:X_train.shape[0]]
X_test = data_complete[X_train.shape[0]:]

h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/zillow_project/zillow_data.h5', 'w')
h5f.create_dataset('X_train', data=X_train)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('X_test', data=X_test)
h5f.close()
