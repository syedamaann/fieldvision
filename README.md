## Overview

This project involves the implementation of a Long Short-Term Memory (LSTM) neural network for time series prediction. The code takes a CSV file containing time-series data with multiple columns, preprocesses the data, builds an LSTM model, trains it, and then uses the trained model to make predictions. Additionally, it provides evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2) for each column.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- `numpy`
- `keras`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `flask`
- `opencv-python`
- `pillow`

You can install these dependencies using the following command:

```bash
pip install numpy keras pandas scikit-learn matplotlib flask opencv-python pillow
```

## Code Structure

The code is divided into two main parts:

1. **LSTM Model Building and Training (`lstm_predict` function):**
   - **Input:** CSV file containing time-series data.
   - **Output:** List of file paths for the generated prediction plots, column names, and error metrics.

2. **Web Application (`app.py`):**
   - A Flask web application for uploading a CSV file and visualizing LSTM predictions.
   - The web app uses the `lstm_predict` function to generate predictions and display the results.

## Usage

1. **Running the Web Application:**
   - Run the Flask web application using the command:

     ```bash
     python app.py
     ```

   - Access the application in your web browser at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

2. **Uploading a CSV File:**
   - Navigate to the "Upload" page.
   - Choose a CSV file containing time-series data.
   - Click the "Upload" button.

3. **Viewing Predictions:**
   - The application will display prediction plots for each column in the uploaded CSV.
   - Error metrics (MAE, MSE, R2) are also provided.

## File Structure

- `app.py`: Main file for the Flask web application.
- `predict.py`: File containing the `lstm_predict` function for LSTM model training and prediction.
- `templates/`: Folder containing HTML templates for the web application.
- `static/`: Folder containing static files (CSS, images) for the web application.
- `output/`: Folder for storing generated prediction plots.

## Dependencies

Ensure you have the required Python libraries installed. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Note

- The LSTM model architecture and parameters are set within the `lstm_predict` function. You may modify them based on your specific use case and dataset.
- The web application utilizes Flask and allows users to upload a CSV file to visualize LSTM predictions.

Feel free to explore and adapt the code for your specific time-series prediction tasks.
