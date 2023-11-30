from flask import Flask, render_template, request
from keras.models import load_model
from pickle import load
import pandas as pd
from datetime import timedelta
import json
import plotly
import plotly.express as px

# Flask web application
app = Flask(__name__)

# load model
model_file = 'bestmodel.h5'
model = load_model(model_file)
scaler_file = 'scaler.pkl'
scaler = load(open(scaler_file, 'rb'))

#route for the root URL ("/")
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_sales = []
    tables = []
    if request.method == 'POST':
        sales_data = request.files.get('sales_data')
        # input data
        df_sales = pd.read_csv(sales_data).set_index('date')
        df_sales.index = pd.to_datetime(df_sales.index)
        # reshape data
        test_values = df_sales.values
        test_values = test_values.reshape(len(test_values), 1)
        # transform using fitted MinMaxScaler
        test_scaled = scaler.transform(test_values)

        # transform to time series data
        def series_to_supervised(data, window=1, lag=1, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data) #buat dataframe
            cols, names = list(), list()
            # input sequence (t-n, ... t-1)
            for i in range(window, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
            # Current timestep (t=0)
            cols.append(df)
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            # put it all together
            agg = pd.concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
            return agg
        
        window = 29
        lag = 60
        test_series = series_to_supervised(test_scaled, window=window, lag=lag)
        
        # Reshape test data
        X_test = test_series.values.reshape((test_series.shape[0], test_series.shape[1], 1))

        # use model to predict
        y_preds = model.predict(X_test)
        y_preds = scaler.inverse_transform(y_preds)
        predicted_sales = y_preds[:,0]
        print('Sales Prediction: ', predicted_sales)

        # visualize prediction
        df_pred = df_sales[window:]
        df_pred.index = df_pred.index + timedelta(days=60)
        df_pred['sales'] = predicted_sales

        # Convert to html table
        test_series = test_series.reset_index(drop=True)
        tables=[test_series.round(2).to_html(classes='data', header="true")]

    return render_template('index.html', TABLES = tables)

if __name__ == '__main__':
    # run the application on a local development server
    app.run(debug=True)
