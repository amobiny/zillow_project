from keras.regularizers import l2
from keras.layers import Dropout
from keras.models import *
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import h5py

h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/zillow_project/zillow_data.h5', 'r')
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
h5f.close()


# Hyper-parameters
EPOCHS = 100
DROP_RATE = 0.3
NUM_HIDDEN = [100, 100]
LEARNING_RATE = 0.001
BATCH_SIZE = 32
REG = 0.0001

# Normalize features
# Test data is *not* used when calculating the mean and std


def normalize(data, mean, std):
    return (data - mean) / std


def denormalize(data, mean, std):
    return (data * std) + mean


x_mean = np.mean(X_train, axis=0)
x_std = np.std(X_train, axis=0)
X_train = normalize(X_train, x_mean, x_std)
X_test = normalize(X_test, x_mean, x_std)

y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_train = normalize(y_train, y_mean, y_std)


num_features = X_train.shape[1]

# Build the model
inputs = Input(shape=(X_train.shape[1],))
inter = Dense(NUM_HIDDEN[0], activation='relu', W_regularizer=l2(REG))(inputs)
inter = Dropout(DROP_RATE)(inter)
inter = Dense(NUM_HIDDEN[1], activation='sigmoid', W_regularizer=l2(REG))(inter)
inter = Dropout(DROP_RATE)(inter)
outputs = Dense(1, W_regularizer=l2(REG))(inter)
model = Model(inputs, outputs)

model.compile(loss='mse',
              optimizer=Adam(lr=LEARNING_RATE),
              metrics=['mae'])
model.summary()
# The patience parameter is the amount of epochs to check for improvement
early_stop = EarlyStopping(monitor='val_loss', patience=20)

# Start Training
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_split=0.2, verbose=1, callbacks=[early_stop])

# Predict
prediction = model.predict(X_test)
prediction = denormalize(prediction, y_mean, y_std)


# Let's plot the error values
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error')
plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label='Val loss')
plt.legend()
plt.show()

print()
