import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. LOAD THE BRAIN (Models and Scalers)
try:
    reg_model = joblib.load('/Users/bipulkumar/Downloads/Data-Set_3/real_estate/regression_model.joblib')
    reg_scaler = joblib.load('/Users/bipulkumar/Downloads/Data-Set_3/real_estate/regression_scaler.joblib')
    clf_model = joblib.load('/Users/bipulkumar/Downloads/Data-Set_3/real_estate/classification_model.joblib')
    clf_scaler = joblib.load('/Users/bipulkumar/Downloads/Data-Set_3/real_estate/classification_scaler.joblib')
except Exception as e:
    st.error(f"Error loading models: {e}")

# 2. DEFINE THE PREDICTION LOGIC
def predict_investment(sqft, bed, bath, age, floor, furnish, neighborhood, dist_city, dist_transport, crime, air, growth, price_sqft, tax, rental):
    # A. Convert Categorical Text to Numbers
    furnish_map = {'Unfurnished': 0, 'Semi-furnished': 1, 'Fully-furnished': 2}
    furnish_val = furnish_map.get(furnish, 1)

    # Neighborhood One-Hot Encoding (IT Hub, Industrial, Residential, Suburban - Downtown is 0 for all)
    n_it = 1 if neighborhood == 'IT Hub' else 0
    n_ind = 1 if neighborhood == 'Industrial' else 0
    n_res = 1 if neighborhood == 'Residential' else 0
    n_sub = 1 if neighborhood == 'Suburban' else 0

    # B. Prepare Input Data in the correct order
    features = {
        'Total_Square_Footage': sqft,
        'Bedrooms': bed,
        'Bathrooms': bath,
        'Age_of_Property': age,
        'Floor_Number': floor,
        'Furnishing_Status': furnish_val,
        'Distance_to_City_Center_km': dist_city,
        'Proximity_to_Public_Transport_km': dist_transport,
        'Crime_Index': crime,
        'Air_Quality_Index': air,
        'Neighborhood_Growth_Rate_%': growth,
        'Price_per_SqFt': price_sqft,
        'Annual_Property_Tax': tax,
        'Estimated_Rental_Yield_%': rental,
        'Neighborhood_IT Hub': n_it,
        'Neighborhood_Industrial': n_ind,
        'Neighborhood_Residential': n_res,
        'Neighborhood_Suburban': n_sub
    }
    
    input_df = pd.DataFrame([features])

    # C. Prediction (Regression)
    scaled_reg = reg_scaler.transform(input_df)
    price = reg_model.predict(scaled_reg)[0]

    # D. Prediction (Classification)
    input_df_clf = input_df.copy()
    input_df_clf['Predicted_Price'] = price
    scaled_clf = clf_scaler.transform(input_df_clf)
    grade = clf_model.predict(scaled_clf)[0]

    return price, grade

# 3. MAIN UI
def main():
    st.title("Real Estate Investment Predictor")
    st.header("Enter Property Details:")

    # Input Fields
    sqft = st.number_input("Total Square Footage", min_value=100)
    bed = st.selectbox("Bedrooms", (1, 2, 3, 4, 5))
    bath = st.selectbox("Bathrooms", (1, 2, 3, 4, 5))
    age = st.number_input("Age of Property (Years)", min_value=0)
    floor = st.number_input("Floor Number", min_value=0)
    furnish = st.selectbox("Furnishing Status", ("Unfurnished", "Semi-furnished", "Fully-furnished"))
    neighborhood = st.selectbox("Neighborhood", ("Downtown", "IT Hub", "Industrial", "Residential", "Suburban"))
    dist_city = st.number_input("Distance to City Center (km)", min_value=0.0)
    dist_transport = st.number_input("Proximity to Public Transport (km)", min_value=0.0)
    crime = st.slider("Crime Index", 0.0, 100.0)
    air = st.slider("Air Quality Index", 0.0, 500.0)
    growth = st.number_input("Neighborhood Growth Rate (%)")
    price_sqft = st.number_input("Price per SqFt ($)", min_value=0)
    tax = st.number_input("Annual Property Tax ($)", min_value=0)
    rental = st.number_input("Estimated Rental Yield (%)", min_value=0.0)

    # THE BUTTON
    if st.button("Analyze Property"):
        price, grade = predict_investment(sqft, bed, bath, age, floor, furnish, neighborhood, dist_city, dist_transport, crime, air, growth, price_sqft, tax, rental)
        
        grade_labels = {0: "Underperformer", 1: "Stable", 2: "High Growth"}
        
        st.success(f"Estimated Market Price: ${price:,.2f}")
        st.info(f"Investment Grade: {grade_labels[grade]}")

if __name__ == "__main__":
    main()
