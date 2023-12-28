# Import necessary libraries
import os
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def split_sequences(sequences, n_steps):
    X, y = [], []
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences) - 1:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def lstm_predict(csv_file_path):
    # Load the dataset
    data = pd.read_csv(csv_file_path, index_col='created_at')
    
    # Handle missing values
    data['P'] = data['P'].replace(np.nan, 20)
    
    # Scale the data
    scaler = MinMaxScaler()
    featured_data = scaler.fit_transform(data.iloc[:, :])
    
    # Reshape the data
    in_seq1 = featured_data[:, 0].reshape((len(featured_data), 1))
    in_seq2 = featured_data[:, 1].reshape((len(featured_data), 1))
    in_seq3 = featured_data[:, 2].reshape((len(featured_data), 1))
    in_seq4 = featured_data[:, 3].reshape((len(featured_data), 1))
    in_seq5 = featured_data[:, 4].reshape((len(featured_data), 1))
    in_seq6 = featured_data[:, 5].reshape((len(featured_data), 1))
    in_seq7 = featured_data[:, 6].reshape((len(featured_data), 1))
    in_seq8 = featured_data[:, 7].reshape((len(featured_data), 1))
    in_seq9 = featured_data[:, 8].reshape((len(featured_data), 1))
    
    # Horizontally stack columns
    dataset = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5, in_seq6, in_seq7, in_seq8, in_seq9))
    
    # Set parameters
    n_steps = 25
    train_rows = int(0.8 * len(dataset))
    
    # Create sequences
    X, y = split_sequences(dataset, n_steps)
    
    # Split into training and testing sets
    x_train, x_test = X[:train_rows, :], X[train_rows:, :]
    y_train, y_test = y[:train_rows], y[train_rows:]
    
    # Build LSTM model
    n_features = X.shape[2]
    model = Sequential()
    n_units1 = 5
    model.add(LSTM(n_units1, activation='tanh', return_sequences=False, input_shape=(n_steps, n_features)))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    n_epochs = 300
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=n_epochs, batch_size=100, verbose=2, shuffle=True)
    
    # Inverse transform and evaluate for each column
    columns = ['P', 'Relative Humidity', 'Temperature', 'EC', 'N', 'P', 'K', 'temp', 'humidity']
    errors = []
    output_file_list = []
    error_names = []

    for i, column_name in enumerate(columns):
        y_pred = model.predict(x_test)
        pred_data = scaler.inverse_transform(y_pred)[:, i]
        actual_data = data.iloc[(train_rows + n_steps):, i].values
        
        MAE = mean_absolute_error(actual_data, pred_data)
        MSE = mean_squared_error(actual_data, pred_data)
        R2 = r2_score(actual_data, pred_data)
        
        errors.append((MAE, MSE, R2))
        error_names.append(column_name)
        
        # Plot the results and save to the output folder
        plt.clf()
        plt.plot(actual_data, label="Actual Data", c="r")
        plt.plot(pred_data, label="Prediction", c="b", linestyle='dotted', linewidth=1)
        plt.title(column_name)
        
        # Generate a unique name for each image
        output_file_name = f"static/output/{column_name}_LSTM_{np.random.randint(1, 100000)}.png"
        output_file_list.append(output_file_name)
        
        # Save the plot
        plt.legend()
        plt.savefig(output_file_name, dpi=600)
        # plt.show()
        plt.clf()

    return output_file_list, error_names, errors

# Create the output folder if it doesn't exist
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

