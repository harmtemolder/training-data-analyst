import numpy as np
import os
import pandas as pd
import shutil
import tensorflow as tf

# Fix "OMP: Error #15: Initializing libiomp5.dylib, but found
# libiomp5.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Start with a fresh output dir each time
OUTPUT_DIR = 'output'
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Read train and test data from CSVs
data_train = pd.read_csv('input/cylinders_train.csv')
data_test = pd.read_csv('input/cylinders_test.csv')

# Define feature columns
feature_columns = [
    tf.feature_column.numeric_column(key='radius', dtype=tf.float64),
    tf.feature_column.numeric_column(key='height', dtype=tf.float64)
]

# Create a linear regressor model
model = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    model_dir=OUTPUT_DIR)

# Train the model
BATCH_SIZE = 4
NUM_EPOCHS = 4

train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=data_train,
    y=data_train['volume'],
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    shuffle=True,
    queue_capacity=1000)

model.train(train_input_fn)

# Test the trained model
test_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=data_test,
    y=data_test['volume'],
    batch_size=BATCH_SIZE,
    shuffle=False,
    queue_capacity=1000)

test_results = model.evaluate(test_input_fn)
print('RMSE on test dataset = {}'.format(np.sqrt(test_results['average_loss'])))
