# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from keras_tuner.tuners import RandomSearch  # updated import

# Ignore warnings
warnings.filterwarnings('ignore')

# Change working directory
os.chdir('C:/Users/Ganesh/OneDrive/Desktop/skill_dev')

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Load Data (CSV files)
train_data = pd.read_csv('C:/Users/Ganesh/OneDrive/Desktop/skill_dev/input/train.csv')
test_data = pd.read_csv('C:/Users/Ganesh/OneDrive/Desktop/skill_dev/input/test.csv')

print(f"Shape of train data: {train_data.shape}")
print(f"Shape of test data: {test_data.shape}")

# Data Overview
pd.set_option("display.max_columns", None)
print(train_data.head())
print(train_data.columns)
print(train_data.describe())
print(train_data['Activity'].unique())

# Plot activity distribution
train_data['Activity'].value_counts().sort_values().plot(kind='bar', color='pink')
plt.title('Activity Distribution')
plt.show()

# Split features and labels
x_train = train_data.iloc[:, :-2]
y_train = train_data.iloc[:, -1]

x_test = test_data.iloc[:, :-2]
y_test = test_data.iloc[:, -1]

print("Shapes after split:")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Normalize features
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Build a simple base model
model = Sequential([
    Dense(units=64, kernel_initializer='normal', activation='sigmoid', input_dim=x_train.shape[1]),
    Dropout(0.2),
    Dense(units=6, kernel_initializer='normal', activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train base model
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

# Define a function for model tuning
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 5)):  # Reduced number of layers for speed
        model.add(layers.Dense(
            units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation_' + str(i), ['relu', 'sigmoid', 'tanh']),
            kernel_initializer=hp.Choice('init_' + str(i), ['uniform', 'normal'])
        ))
    model.add(Dropout(0.2))
    model.add(layers.Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Random Search Tuning
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='project',
    project_name='Human_activity_recognition'
)

tuner.search_space_summary()

# Run tuner search
tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]

# Train best model further
early_stop = EarlyStopping(monitor='accuracy', patience=3)

mo_fitt = best_model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test), callbacks=[early_stop])

# Model Summary
best_model.summary()

# Plot Accuracy and Loss
accuracy = mo_fitt.history['accuracy']
val_accuracy = mo_fitt.history['val_accuracy']
loss = mo_fitt.history['loss']
val_loss = mo_fitt.history['val_loss']

epochs_range = range(len(accuracy))

plt.figure(figsize=(15, 7))

plt.subplot(2, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss')

plt.tight_layout()
plt.show()

# ===== Final Evaluation and Accuracy =====
test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
