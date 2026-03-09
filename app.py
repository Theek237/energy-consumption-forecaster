import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ===== Page Config =====
st.set_page_config(
    page_title="Energy Consumption Forecaster",
    page_icon="⚡",
    layout="wide"
)

# ===== Load Model & Artifacts =====
@st.cache_resource
def load_artifacts():
    from tensorflow.keras.models import load_model
    model = load_model('energy_model.h5')
    scaler_X = joblib.load('scaler_X.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler_X, feature_names

model, scaler_X, feature_names = load_artifacts()

# ===== Header =====
st.title("⚡ Household Energy Consumption Forecaster")
st.markdown("Predict household appliance energy usage using an Artificial Neural Network trained on the UCI Appliances Energy dataset.")
st.markdown("---")

# ===== Sidebar Inputs =====
st.sidebar.header("🔧 Input Parameters")

# Time
st.sidebar.subheader("⏰ Time")
selected_date = st.sidebar.date_input("Date", datetime.now())
selected_hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

# Indoor Temperatures
st.sidebar.subheader("🌡️ Indoor Temperature (°C)")
t1 = st.sidebar.slider("T1 - Kitchen", 10.0, 30.0, 21.0, 0.5)
t2 = st.sidebar.slider("T2 - Living Room", 10.0, 30.0, 20.0, 0.5)
t3 = st.sidebar.slider("T3 - Laundry", 10.0, 30.0, 22.0, 0.5)
t4 = st.sidebar.slider("T4 - Office", 10.0, 30.0, 20.0, 0.5)
t5 = st.sidebar.slider("T5 - Bathroom", 10.0, 30.0, 20.0, 0.5)
t7 = st.sidebar.slider("T7 - Ironing Room", 10.0, 30.0, 20.0, 0.5)
t8 = st.sidebar.slider("T8 - Teenager Room", 10.0, 30.0, 20.0, 0.5)
t9 = st.sidebar.slider("T9 - Parents Room", 10.0, 30.0, 20.0, 0.5)

# Outdoor
st.sidebar.subheader("🌤️ Outdoor Conditions")
t6 = st.sidebar.slider("T6 - Outside Temp (°C)", -10.0, 30.0, 10.0, 0.5)
t_out = st.sidebar.slider("T_out (°C)", -10.0, 30.0, 10.0, 0.5)
tdewpoint = st.sidebar.slider("Dewpoint (°C)", -10.0, 20.0, 5.0, 0.5)
press = st.sidebar.slider("Pressure (mm Hg)", 725.0, 775.0, 755.0, 1.0)
windspeed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 15.0, 4.0, 0.5)
visibility = st.sidebar.slider("Visibility (km)", 1.0, 70.0, 40.0, 1.0)

# Humidity
st.sidebar.subheader("💧 Humidity (%)")
rh_1 = st.sidebar.slider("RH_1 - Kitchen", 20, 70, 40)
rh_2 = st.sidebar.slider("RH_2 - Living Room", 20, 70, 40)
rh_3 = st.sidebar.slider("RH_3 - Laundry", 20, 60, 39)
rh_4 = st.sidebar.slider("RH_4 - Office", 20, 60, 39)
rh_5 = st.sidebar.slider("RH_5 - Bathroom", 20, 60, 39)
rh_6 = st.sidebar.slider("RH_6 - Outside", 20, 100, 50)
rh_7 = st.sidebar.slider("RH_7 - Ironing Room", 20, 60, 40)
rh_8 = st.sidebar.slider("RH_8 - Teenager Room", 20, 60, 40)
rh_9 = st.sidebar.slider("RH_9 - Parents Room", 20, 60, 42)
rh_out = st.sidebar.slider("RH_out", 20, 100, 80)

# Lights
lights = st.sidebar.slider("💡 Lights (Wh)", 0, 70, 0)


# ===== Build Feature Vector =====
def build_features(hour, date, t1, t2, t3, t4, t5, t6, t7, t8, t9, t_out,
                   rh_1, rh_2, rh_3, rh_4, rh_5, rh_6, rh_7, rh_8, rh_9, rh_out,
                   tdewpoint, press, windspeed, visibility, lights):

    day_of_week = date.weekday()
    month = date.month
    is_weekend = 1 if day_of_week >= 5 else 0
    nsm = hour * 3600

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    t_diff_kitchen = t1 - t6
    t_diff_living = t2 - t6
    t_diff_laundry = t3 - t6
    t_indoor_avg = np.mean([t1, t2, t3, t4, t5, t7, t8, t9])
    rh_indoor_avg = np.mean([rh_1, rh_2, rh_3, rh_4, rh_5, rh_7, rh_8, rh_9])
    rh_diff = rh_indoor_avg - rh_6

    input_dict = {
        'lights': lights,
        'T1': t1, 'RH_1': rh_1, 'T2': t2, 'RH_2': rh_2,
        'T3': t3, 'RH_3': rh_3, 'T4': t4, 'RH_4': rh_4,
        'T5': t5, 'RH_5': rh_5, 'T6': t6, 'RH_6': rh_6,
        'T7': t7, 'RH_7': rh_7, 'T8': t8, 'RH_8': rh_8,
        'T9': t9, 'RH_9': rh_9,
        'T_out': t_out, 'Press_mm_hg': press,
        'RH_out': rh_out, 'Windspeed': windspeed,
        'Visibility': visibility, 'Tdewpoint': tdewpoint,
        'NSM': nsm,
        'hour': hour, 'day_of_week': day_of_week,
        'month': month, 'is_weekend': is_weekend,
        'hour_sin': hour_sin, 'hour_cos': hour_cos,
        'day_sin': day_sin, 'day_cos': day_cos,
        'month_sin': month_sin, 'month_cos': month_cos,
        'T_diff_kitchen': t_diff_kitchen,
        'T_diff_living': t_diff_living,
        'T_diff_laundry': t_diff_laundry,
        'T_indoor_avg': t_indoor_avg,
        'RH_indoor_avg': rh_indoor_avg,
        'RH_diff': rh_diff,
    }

    input_df = pd.DataFrame([input_dict])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[feature_names]


# ===== Predict =====
if st.button("🔮 Predict Energy Consumption", type="primary", use_container_width=True):

    input_df = build_features(
        selected_hour, selected_date,
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t_out,
        rh_1, rh_2, rh_3, rh_4, rh_5, rh_6, rh_7, rh_8, rh_9, rh_out,
        tdewpoint, press, windspeed, visibility, lights
    )
    input_scaled = scaler_X.transform(input_df)
    pred_log = model.predict(input_scaled, verbose=0).flatten()[0]
    prediction = max(0, np.expm1(pred_log))

    st.markdown("---")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⚡ Predicted Energy", f"{prediction:.1f} Wh")
    with col2:
        daily_est = prediction * 144  # 10-min intervals per day
        st.metric("📊 Daily Estimate", f"{daily_est/1000:.1f} kWh")
    with col3:
        indoor_avg = np.mean([t1, t2, t3, t4, t5, t7, t8, t9])
        st.metric("🌡️ Indoor Avg", f"{indoor_avg:.1f} °C")
    with col4:
        st.metric("🌡️ In-Out Diff", f"{indoor_avg - t6:.1f} °C")

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Energy Consumption (Wh)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1},
            'bar': {'color': "#FF5722"},
            'steps': [
                {'range': [0, 100], 'color': '#C8E6C9'},
                {'range': [100, 200], 'color': '#FFF9C4'},
                {'range': [200, 350], 'color': '#FFE0B2'},
                {'range': [350, 500], 'color': '#FFCDD2'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 350
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 24-hour forecast
    st.subheader("📈 24-Hour Energy Forecast")
    hourly_preds = []
    for h in range(24):
        h_df = build_features(
            h, selected_date,
            t1, t2, t3, t4, t5, t6, t7, t8, t9, t_out,
            rh_1, rh_2, rh_3, rh_4, rh_5, rh_6, rh_7, rh_8, rh_9, rh_out,
            tdewpoint, press, windspeed, visibility, lights
        )
        h_scaled = scaler_X.transform(h_df)
        h_pred = model.predict(h_scaled, verbose=0).flatten()[0]
        hourly_preds.append(max(0, np.expm1(h_pred)))

    hourly_df = pd.DataFrame({
        'Hour': [f"{h:02d}:00" for h in range(24)],
        'Energy (Wh)': hourly_preds
    })

    fig_bar = px.bar(
        hourly_df, x='Hour', y='Energy (Wh)',
        color='Energy (Wh)',
        color_continuous_scale='RdYlGn_r',
        title='Predicted Energy by Hour (same conditions)'
    )
    fig_bar.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Peak Hour", f"{hourly_df.loc[hourly_df['Energy (Wh)'].idxmax(), 'Hour']}")
    with col2:
        st.metric("Peak Energy", f"{max(hourly_preds):.1f} Wh")
    with col3:
        st.metric("Daily Total", f"{sum(hourly_preds)/1000:.2f} kWh")

# ===== Footer =====
st.markdown("---")
st.markdown(
    "**Model:** ANN (512→256→128→64→32→1) with BatchNorm & Dropout | "
    "**Dataset:** UCI Appliances Energy (19,735 samples) | "
    "**R² Score:** 0.45 | "
    "**Built with:** TensorFlow, Streamlit, Plotly"
)
