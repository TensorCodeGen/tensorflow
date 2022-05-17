
from numpy import loadtxt
import numpy as np

import tensorflow as tf


from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Sequential

## Explicitly disable GPU to force CPU usage
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['AUTOGRAPH_VERBOSITY'] = '2'

tf.get_logger().setLevel('ERROR')


TENSOR_SIZE = 64
OUTPUT_SIZE = 8
BATCH_SIZE = 4

num_data = int(TENSOR_SIZE * 10)
X = np.random.rand(num_data,TENSOR_SIZE)
y = np.random.rand(num_data,OUTPUT_SIZE)


model = Sequential()
model.add(Dense(TENSOR_SIZE, input_dim = TENSOR_SIZE , activation="tanh"))
model.add(Dense(32, activation="tanh"))
model.add(BatchNormalization())
model.add(Dense(16, activation="tanh"))
model.add(Dense(OUTPUT_SIZE, input_dim = TENSOR_SIZE , activation="tanh"))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


model.fit(X, y, epochs=1, batch_size=BATCH_SIZE)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
