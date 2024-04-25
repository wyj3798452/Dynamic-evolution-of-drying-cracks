
import math
import shap
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


dataset = read_csv('Primitive_crack.csv', header=0, index_col=0)
values = dataset.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 10, 1)
reframed.drop(reframed.columns[[62, 63, 64, 65]], axis=1, inplace=True)
print(reframed)
values = reframed.values
train = values[:5760]
test = values[5760:]
print(train)
train_X, train_y = train[:, :-2], train[:, -2:]
print(train_y)
test_X, test_y = test[:, :-2], test[:, -2:]
train_X = train_X.reshape((train_X.shape[0], 10, 6))
test_X = test_X.reshape((test_X.shape[0], 10, 6))



def create_model(output_dim=100, dropout_rate=0.1, optimizer='adam', learning_rate=0.01):
    if optimizer == 'adam':
        optimizer = Adam(lr=learning_rate)
    model = Sequential()
    model.add(LSTM(output_dim, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2))
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
def plot_training_test_loss(train_loss, val_loss):
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ['#1f77b4', '#ff7f0e']
    ax.plot(train_loss, label='Training Loss', color=colors[0], linestyle='-', marker='o', markersize=5, linewidth=2)

    ax.plot(val_loss, label='Validation Loss', color=colors[1], linestyle='-', marker='o', markersize=5, linewidth=2)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title('Training and Test Loss over Epochs', fontsize=16)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend()

    fig.tight_layout()
    plt.savefig('training_test_loss.png', dpi=300)
    plt.show()
plot_training_test_loss(train_loss, val_loss)


predicted_train_y = best_model.predict(train_X)
predicted_test_y = best_model.predict(test_X)
def plot_actual_vs_predicted(train_y, predicted_train_y, test_y, predicted_test_y):
    combined_x_train = list(range(len(train_y)))
    combined_x_test = list(range(len(train_y), len(train_y) + len(test_y)))

    plt.figure(figsize=(15, 6), dpi=300)

    plt.plot(combined_x_train, train_y[:, 0], label='Actual (Train, Crack Ratio)', color='#3498db', linestyle='-',
             marker='o', markersize=1, linewidth=0.5)
    plt.plot(combined_x_train, predicted_train_y[:, 0], label='Predicted (Train, Crack Ratio)', color='#e74c3c',
             linestyle='--', marker='x', markersize=1, linewidth=0.5)

    plt.plot(combined_x_test, test_y[:, 0], label='Actual (Test, Crack Ratio)', color='#2ecc71', linestyle='-',
             marker='o', markersize=1, linewidth=0.5)
    plt.plot(combined_x_test, predicted_test_y[:, 0], label='Predicted (Test, Crack Ratio)', color='#f39c12',
             linestyle='--', marker='x', markersize=1, linewidth=0.5)

    plt.plot(combined_x_train, train_y[:, 1], label='Actual (Train, Crack Width)', color='#1f77b4', linestyle='-',
             marker='o', markersize=1, linewidth=0.5)
    plt.plot(combined_x_train, predicted_train_y[:, 1], label='Predicted (Train, Crack Width)', color='#ff7f0e',
             linestyle='--', marker='x', markersize=1, linewidth=0.5)

    plt.plot(combined_x_test, test_y[:, 1], label='Actual (Test, Crack Width)', color='#d62728', linestyle='-',
             marker='o', markersize=1, linewidth=0.5)
    plt.plot(combined_x_test, predicted_test_y[:, 1], label='Predicted (Test, Crack Width)', color='#9467bd',
             linestyle='--', marker='x', markersize=1, linewidth=0.5)

    plt.title('Actual vs Predicted', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Value', fontsize=14)

    plt.legend(loc='upper left', fontsize=10)

    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    plt.savefig('Actual_vs_Predicted.png', dpi=300)

    plt.show()
plot_actual_vs_predicted(train_y, predicted_train_y, test_y, predicted_test_y)


def export_predictions_to_csv(true_values, predicted_values, output_file):
    columns = []
    for i in range(true_values.shape[1]):
        columns.append(f'True Value (Variable {i + 1})')
        columns.append(f'Predicted Value (Variable {i + 1})')

    df = pd.DataFrame(columns=columns)
    for i in range(true_values.shape[0]):
        row = []
        for j in range(true_values.shape[1]):
            row.append(true_values[i, j])
            row.append(predicted_values[i, j])
        df.loc[len(df)] = row

    df.to_csv(output_file, index=False)

export_predictions_to_csv(train_y, predicted_train_y, 'train_true_predicted.csv')
export_predictions_to_csv(test_y, predicted_test_y, 'test_true_predicted.csv')


def plot_comparison(true_values, predicted_values, title='Model Performance', save_path=None):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    rmse = np.sqrt(mse)

    plt.figure(figsize=(10, 8))

    plt.scatter(true_values[:, 0], predicted_values[:, 0], color='gray', marker='*', label='True vs Predicted (Feature 1)', alpha=0.6)
    plt.scatter(true_values[:, 1], predicted_values[:, 1], color='blue', marker='o', label='True vs Predicted (Feature 2)', alpha=0.6)

    plt.plot([min(true_values[:, 0]), max(true_values[:, 0])], [min(true_values[:, 0]), max(true_values[:, 0])], color='red', linestyle='--', label='1:1 Reference Line (Feature 1)', linewidth=2)
    plt.plot([min(true_values[:, 1]), max(true_values[:, 1])], [min(true_values[:, 1]), max(true_values[:, 1])], color='green', linestyle='--', label='1:1 Reference Line (Feature 2)', linewidth=2)

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

plot_comparison(train_y, predicted_train_y, title='Model Performance on Training Data', save_path='train_comparison_plot.png')
plot_comparison(test_y, predicted_test_y, title='Model Performance on Test Data', save_path='test_comparison_plot.png')
