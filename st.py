import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
model = joblib.load(open("hybrid_model.pkl", "rb"))

def predict_with_selected_features(pktcount, src_10, src_1, bytecount, src_12, src_2, model):
    # Create dummy values for the remaining features
    dummy_values = [0] * 65  # Assuming the remaining 65 features are numerical
    
    # Combine the provided features with dummy values
    all_features = [pktcount, src_10, src_1, bytecount, src_12, src_2] + dummy_values
    
    # Reshape the data to match the format expected by the model
    input_data = np.array(all_features).reshape(1, -1)
    
    # Predict the outcome using the provided model
    prediction = model.predict(input_data)
    
    return prediction[0]  # Return the predicted outcome

def main():
    st.sidebar.title("Cybersecurity")
    st.sidebar.image("cybersecurity_logo.jpg", use_column_width=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quote of the Day")
    st.sidebar.markdown("#### \"Cybersecurity is a mindset\"")

    st.title("Category of BotNet Attack Prediction:globe_with_meridians:")

    # Input form
    avg_dur = st.number_input("Average duration of packet")
    stddev_dur = st.number_input("Standard deviation of packet")
    min_dur = st.number_input("Minimum duration of packet")
    max_dur = st.number_input("Maximum duration of packet")
    srate = st.number_input("Source to destination speed")
    drate = st.number_input("Destination to source speed")

    # Predict button
    if st.button("Predict"):
        # Make prediction
        with st.spinner('Predicting...'):
            output = predict_with_selected_features(avg_dur, stddev_dur, min_dur, max_dur, srate, drate, model)

        # Show prediction result
        if output == 1:
            st.error("It's a BotNet Attack!")
        else:
            st.success("No Attacks. Normal packet transfer.")

if __name__ == "__main__":
    main()
