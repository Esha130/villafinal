import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import streamlit as st
import io
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Load data
file_path = "official_dataset2.xlsx"  # Update the file path
sheets_dict = pd.read_excel(file_path, sheet_name=None)
result = sheets_dict.get('result')
availability = sheets_dict.get('availability')

# Preprocess data
result['date'] = pd.to_datetime(result['date'], errors='coerce').dt.date
availability['date'] = pd.to_datetime(availability['date'], errors='coerce').dt.date

# List of holidays
holidays = [
    '2024-01-26', '2024-03-25', '2024-03-29', '2024-04-11', '2024-04-17', '2024-04-21', '2024-05-23',
    '2024-06-17', '2024-07-17', '2024-08-15', '2024-08-26', '2024-09-16', '2024-10-02', '2024-10-12',
    '2024-10-31', '2024-11-15', '2024-12-25', '2024-12-31', '2025-01-01', '2025-01-14', '2025-01-26',
    '2025-03-14', '2025-03-30'
]
holidays = pd.to_datetime(holidays).date

# Map categorical variables to numeric values
villa_mapping = {v: i for i, v in enumerate(result['villa'].unique())}
city_mapping = {c: i for i, c in enumerate(result['city'].unique())}

result['villa_encoded'] = result['villa'].map(villa_mapping)
result['city_encoded'] = result['city'].map(city_mapping)
result['SEASONS'] = result['SEASONS'].astype('category').cat.codes

# Add weekend and holiday features to the dataset
result['is_weekend'] = result['date'].apply(lambda x: x.weekday() >= 4)  # Saturday (5), Sunday (6)
result['is_holiday'] = result['date'].apply(lambda x: x in holidays)

# Train models with normalized features
features = [
    'villa_encoded', 'city_encoded', 'SEASONS', 'Premiumness',
    'total_capacity', 'baths_count', 'bedroom_count',
    'Cafeology Paraphernalia', 'Bonfire', 'Golf Club Set',
    'Private Pool', 'Swimming Pool(Private)', 'Jacuzzi',
    'is_weekend', 'is_holiday'
]
target = 'price'

scaler = StandardScaler()
X = scaler.fit_transform(result[features])
y = result[target]

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

xgb_model = XGBRegressor(random_state=42, verbosity=0)
xgb_model.fit(X, y)

# Helper functions
def get_previous_year_price(selected_date, selected_villa):
    previous_year_date = selected_date - timedelta(days=366)
    previous_year_data = result[
        (result['villa'] == selected_villa) & 
        (result['date'] == previous_year_date)
    ]
    return previous_year_data, previous_year_date

def get_previous_year_avg_price(selected_villa, selected_date):
    previous_year = selected_date.year - 1
    previous_year_data = result[
        (result['villa'] == selected_villa) & 
        (result['date'].apply(lambda x: x.year) == previous_year)
    ]
    if not previous_year_data.empty:
        return previous_year_data['price'].mean()
    return None

def is_villa_available(villa, selected_date):
    availability_data = availability[
        (availability['villa'] == villa) & 
        (availability['date'] == selected_date)
    ]
    if not availability_data.empty:
        return availability_data['status'].iloc[0].lower() == 'available'
    return False

# Streamlit App
st.title("Villa Price Prediction")

# User input
selected_city = st.selectbox("Select a city", result['city'].unique())
filtered_villas = result[result['city'] == selected_city]['villa'].unique()
selected_villas = st.multiselect("Select villas", filtered_villas)
multiplier = st.number_input("Enter Input for Multiplier", 0.00, None)

if selected_villas:
    date_range = pd.date_range(datetime.today(), '2025-01-01').date

    predictions = []
    for selected_date in date_range:
        for villa in selected_villas:
            available = is_villa_available(villa, selected_date)

            previous_year_data, prev_year_date = get_previous_year_price(selected_date, villa)

            if not previous_year_data.empty:
                prev_year_price = previous_year_data['price'].iloc[0]
            else:
                prev_year_price = get_previous_year_avg_price(villa, selected_date)
                if prev_year_price is None:
                    feature_data = {
                        'villa_encoded': villa_mapping[villa],
                        'city_encoded': city_mapping[selected_city],
                        'SEASONS': result[result['villa'] == villa]['SEASONS'].iloc[0],
                        'Premiumness': result[result['villa'] == villa]['Premiumness'].iloc[0],
                        'total_capacity': result[result['villa'] == villa]['total_capacity'].iloc[0],
                        'baths_count': result[result['villa'] == villa]['baths_count'].iloc[0],
                        'bedroom_count': result[result['villa'] == villa]['bedroom_count'].iloc[0],
                        'Cafeology Paraphernalia': result[result['villa'] == villa]['Cafeology Paraphernalia'].iloc[0],
                        'Bonfire': result[result['villa'] == villa]['Bonfire'].iloc[0],
                        'Golf Club Set': result[result['villa'] == villa]['Golf Club Set'].iloc[0],
                        'Private Pool': result[result['villa'] == villa]['Private Pool'].iloc[0],
                        'Swimming Pool(Private)': result[result['villa'] == villa]['Swimming Pool(Private)'].iloc[0],
                        'Jacuzzi': result[result['villa'] == villa]['Jacuzzi'].iloc[0],
                        'is_weekend': selected_date.weekday() >= 4,
                        'is_holiday': selected_date in holidays
                    }

                    feature_df = pd.DataFrame([feature_data])
                    feature_df_scaled = scaler.transform(feature_df)

                    rf_price_model = rf_model.predict(feature_df_scaled)[0]
                    xgb_price_model = xgb_model.predict(feature_df_scaled)[0]

                    prev_year_price = (rf_price_model + xgb_price_model) / 2

            random_variation_rf = np.random.uniform(0.98, 1.02)
            random_variation_xgb = np.random.uniform(0.98, 1.02)
            rf_price = max(prev_year_price * 0.9, prev_year_price * random_variation_rf)
            xgb_price = max(prev_year_price * 0.9, prev_year_price * random_variation_xgb)

            predictions.append({
                'Villa': villa,
                'Availability': 'Available' if available else 'Unavailable',
                'Date': selected_date.strftime('%Y-%m-%d'),
                'Previous Year Date': prev_year_date.strftime('%Y-%m-%d') if not previous_year_data.empty else "Model-Based",
                'Previous Year Price': int(round(prev_year_price, 0)),
                'Random Forest Price': int(round(rf_price, 0)),
                'XGBoost Price': int(round(xgb_price, 0)),
                'Adjusted Price (with multiplier)': int(round((rf_price + xgb_price) / 2 * multiplier, 0))
            })

    predictions_df = pd.DataFrame(predictions)
    st.write(f"Price Predictions for {selected_city} from {datetime.today().strftime('%Y-%m-%d')} to 2024-12-31")
    st.dataframe(predictions_df)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        predictions_df.to_excel(writer, index=False, sheet_name='Predictions')

    buffer.seek(0)
    st.download_button(
        label="Download data as Excel",
        data=buffer,
        file_name="villa_price_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
