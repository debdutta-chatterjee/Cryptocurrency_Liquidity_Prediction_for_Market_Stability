import streamlit as st
import pandas as pd
import joblib 

try:
    st.title('Cryptocurrency liquidity prediction for market stability')

    # Creating number inputs for each field
    price = st.number_input("Price", value=1.256500)
    hour_1 = st.number_input("1h Change", value=1.517170)
    hour_24 = st.number_input("24h Change", value=3.268147)
    day_7 = st.number_input("7d Change", value=1.570604)
    volume_24h = st.number_input("24h Volume", value=1.529558)
    market_cap = st.number_input("Market Cap", value=1.588874)
    moving_averages = st.number_input("Moving Averages", value=1.269748)
    volatility = st.number_input("Volatility", value=2.680702)

    # Display entered values
    st.write("Entered Values:")

    df = pd.DataFrame(
        {
        "price": [price],
        "1h": [hour_1],
        "24h": [hour_24],
        "7d": [day_7],
        "24h_volume": [volume_24h],
        "mkt_cap": [market_cap],
        "moving_averages": [moving_averages],
        "volatility": [volatility]
        }
    )

    if st.button('Submit'):
        st.subheader("Entered Data")
        st.dataframe(df)

        model = joblib.load('deliverables/final_model.pkl')
        scaler = joblib.load('deliverables/scaler_model.pkl')

        X_test = scaler.transform(df)
        y_pred = model.predict(X_test)

        st.subheader("Predicted Liquidity Ratio")
        st.write(y_pred)


except Exception as e:
    st.write(f"Exception {e}")