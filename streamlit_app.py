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
    '2025-01-01', '2025-01-14', '2025-01-26',
    '2025-03-01', '2025-03-17', '2025-03-30',
    '2025-04-14', '2025-04-18', '2025-05-01',
    '2025-05-12', '2025-06-06', '2025-08-15',
    '2025-08-16', '2025-08-30', '2025-10-02',
    '2025-10-07', '2025-10-11', '2025-10-20',
    '2025-10-29', '2025-10-30', '2025-10-31',
    '2025-11-02', '2025-12-25'
]

holidays = pd.to_datetime(holidays).date

# Map categorical variables to numeric values
villa_mapping = {v: i for i, v in enumerate(result['villa'].unique())}
city_mapping = {c: i for i, c in enumerate(result['city'].unique())}

result['villa_encoded'] = result['villa'].map(villa_mapping)
result['city_encoded'] = result['city'].map(city_mapping)
result['SEASONS'] = result['SEASONS'].astype('category').cat.codes

# Add weekend and holiday features
result['is_weekend'] = result['date'].apply(lambda x: x.weekday() >= 4)  # Saturday (5), Sunday (6)
result['is_holiday'] = result['date'].apply(lambda x: x in holidays)

# Train models
features = [
    'villa_encoded', 'city_encoded', 'SEASONS', 'Premiumness',
    'total_capacity', 'is_weekend', 'is_holiday'
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

# User Inputs
selected_city = st.selectbox("Select a city", result['city'].unique())
filtered_villas = result[result['city'] == selected_city]['villa'].unique()
selected_villas = st.multiselect("Select villas", filtered_villas)
multiplier = st.number_input("Enter Input for Multiplier", 0.00, None)

# New: Date Picker
start_date, end_date = st.date_input(
    "Select date range",
    value=(datetime.today().date(), datetime(2025, 12, 31).date())
)

if selected_villas and start_date and end_date and start_date <= end_date:
    date_range = pd.date_range(start=start_date, end=end_date)

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
    st.write(f"Price Predictions for {selected_city} from {start_date} to {end_date}")
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
else:
    st.warning("Please select valid villas and a proper date range.")
