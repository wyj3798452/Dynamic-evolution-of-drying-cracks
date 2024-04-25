
import math
import numpy as np
import pandas as pd
from matplotlib import cm
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from tensorflow.keras.callbacks import Callback
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import LSTM,Dropout,Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from sklearn.metrics import mean_squared_error, mean_absolute_error,make_scorer, r2_score


def series_to_supervised(data, n_in=10, n_out=1, dropnan=True):
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# Load dataset
dataset = read_csv('crack_ratio_single.csv', header=0, index_col=0)
values = dataset.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 10, 1)
print(reframed)
values = reframed.values
train = values[:5760]
test = values[5760:]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
train_X = train_X.reshape((train_X.shape[0], 10, 1))
test_X = test_X.reshape((test_X.shape[0], 10, 1))


def create_model(output_dim=100, dropout_rate=0.1, optimizer='adam', learning_rate=0.01):
    if optimizer == 'adam':
        optimizer = Adam(lr=learning_rate)
    model = Sequential()
    model.add(LSTM(output_dim, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

model = KerasRegressor(build_fn=create_model, verbose=0)

param_grid = {
    'output_dim': [100],
    'batch_size': [32],
    'epochs': [150],
    'dropout_rate': [0.1],
    'optimizer': ['adam'],
    'learning_rate': [0.01]
}

tscv = TimeSeriesSplit(n_splits=5)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, verbose=2, n_jobs=1)
grid_result = grid.fit(train_X, train_y)

print("Best parameters found: ", grid_result.best_params_)
print("Best negative mean squared error: ", -grid_result.best_score_)

best_model = grid_result.best_estimator_
history = best_model.fit(train_X, train_y, validation_data=(test_X, test_y), verbose=2, shuffle=False)


train_loss = history.history['loss']
val_loss = history.history['val_loss']
def plot_training_validation_loss(train_loss, val_loss):
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ['#1f77b4', '#ff7f0e']

    ax.plot(train_loss, label='Training Loss', color=colors[0], linestyle='-', marker='o', markersize=5, linewidth=2)

    ax.plot(val_loss, label='Validation Loss', color=colors[1], linestyle='-', marker='o', markersize=5, linewidth=2)

    ax.grid(True, linestyle='--', alpha=0.6)

    ax.set_title('Training and Validation Loss over Epochs', fontsize=16)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)

    ax.legend()

    fig.tight_layout()

    plt.savefig('Training_and_Validation_Loss_over_Epochs.png', dpi=300)

    plt.show()
plot_training_validation_loss(train_loss, val_loss)

predicted_train_y = best_model.predict(train_X)
predicted_test_y = best_model.predict(test_X)
def plot_actual_vs_predicted(train_y, predicted_train_y, test_y, predicted_test_y):
    # Combine the x-coordinates of the training and test sets
    combined_x = list(range(len(train_y))) + list(range(len(train_y), len(train_y) + len(test_y)))

    plt.figure(figsize=(15, 6), dpi=300)

    plt.plot(combined_x[:len(train_y)], train_y, label='Actual (Train)', color='#3498db', linestyle='-', marker='o',
             markersize=1, linewidth=0.5)
    plt.plot(combined_x[:len(train_y)], predicted_train_y, label='Predicted (Train)', color='#e74c3c',
             linestyle='--', marker='x', markersize=1, linewidth=0.5)

    plt.plot(combined_x[len(train_y):], test_y, label='Actual (Test)', color='#2ecc71', linestyle='-', marker='o',
             markersize=1, linewidth=0.5)
    plt.plot(combined_x[len(train_y):], predicted_test_y, label='Predicted (Test)', color='#f39c12', linestyle='--',
             marker='x', markersize=1, linewidth=0.5)

    # Set the title and axis labels
    plt.title('Actual vs Predicted', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Crack Ratio', fontsize=14)

    plt.legend(loc='upper left', fontsize=10)

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    plt.savefig('actual_vs_predicted.png', dpi=300)

    plt.show()
plot_actual_vs_predicted(train_y, predicted_train_y, test_y, predicted_test_y)


def export_predictions(train_y, predicted_train_y, test_y, predicted_test_y):
    # Convert the predicted values and actual values for training set to DataFrame
    df_train = pd.DataFrame({
        'Actual (Train)': train_y,
        'Predicted (Train)': predicted_train_y
    })

    # Convert the predicted values and actual values for test set to DataFrame
    df_test = pd.DataFrame({
        'Actual (Test)': test_y,
        'Predicted (Test)': predicted_test_y
    })

    # Export DataFrames to CSV files
    df_train.to_csv('train_predictions.csv', index=False)
    df_test.to_csv('test_predictions.csv', index=False)
export_predictions(train_y, predicted_train_y, test_y, predicted_test_y)


def plot_comparison(true_values, predicted_values, title='Model Performance', save_path=None):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    rmse = np.sqrt(mse)

    plt.figure(figsize=(10, 8))
    plt.scatter(true_values, predicted_values, color='gray', marker='*', label='True vs Predicted', alpha=0.6)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linestyle='--', label='1:1 Reference Line', linewidth=2)
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)


    textstr = '\n'.join((
        f'MSE = {mse:.3f}',
        f'MAE = {mae:.3f}',
        f'R2 Score = {r2:.3f}',
        f'RMSE = {rmse:.3f}'))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10, horizontalalignment='right', verticalalignment='bottom', bbox=props)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()


    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

plot_comparison(train_y, predicted_train_y, title='Train Set Performance', save_path='train_set_performance.png')
plot_comparison(test_y, predicted_test_y, title='Test Set Performance', save_path='test_set_performance.png')