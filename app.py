import streamlit as st
import pandas as pd
import joblib

# Load data and drop duplicates + unnecessary columns
df = pd.read_csv('diamonds.csv')
df = df.drop_duplicates()

# Drop 'Unnamed: 0' if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Drop target column 'price'
X = df.drop(columns=['price'])

# Load preprocessor and model
model = joblib.load('DTR_Grid.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title("💎 Diamond Price Predictor")

# Row index input
row_index = st.number_input(f"Enter a diamond row index (0 to {len(X)-1}):", step=1, format="%d")

# Prediction
if 0 <= row_index < len(X):
    st.subheader("🔍 Selected Diamond Features")
    selected_row = X.iloc[[int(row_index)]]
    st.write(selected_row)

    if st.button("🔮 Predict Price"):
        transformed_input = preprocessor.transform(selected_row)
        prediction = model.predict(transformed_input)[0]
        st.success(f"💰 Predicted Price: ${prediction:,.2f}")
else:
    st.warning("⚠️ Please enter a valid row index.")
