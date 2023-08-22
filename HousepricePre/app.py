
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


st.title('House Price Prediction')

@st.cache_resource()
def load_data():
    data = pd.read_csv('Cleaned_data.csv')
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    return data

data = load_data()

X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = st.sidebar.selectbox('Select Model', ['Linear Regression', 'Lasso', 'Ridge'])

if model == 'Linear Regression':
    reg = LinearRegression()
elif model == 'Lasso':
    reg = Lasso()
else:
    reg = Ridge()

column_trans = make_column_transformer((OneHotEncoder(sparse_output=False), ['location']),
                                       remainder='passthrough')

scaler = StandardScaler()

pipe = make_pipeline(column_trans, scaler, reg)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

r2 = r2_score(y_test, y_pred)

st.write(f'r2 score of {model} model: {r2:.2f}')

input_data = []

for col in X.columns:
    if col == 'location':
        options = X['location'].unique()
        input_value = st.selectbox(col, options)
    else:
        input_value = st.number_input(col, step=1)

    input_data.append(input_value)

input_df = pd.DataFrame([input_data], columns=X.columns)

predicted_price = pipe.predict(input_df)

if st.button('Predict'):
    st.write(f"Predicted price is {predicted_price[0]:.2f} lakhs")
