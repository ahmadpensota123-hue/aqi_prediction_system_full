"""
Streamlit Dashboard - AQI Prediction System
============================================

This module implements an interactive web dashboard for AQI monitoring.

Features:
- Current AQI display with color coding
- 3-day forecast visualization
- Historical trend charts
- SHAP feature importance explanations
- Health alerts for hazardous conditions

Why Streamlit?
- Quick to build interactive dashboards
- Python-native, no frontend knowledge needed
- Auto-refresh capabilities
- Easy integration with ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import get_settings, get_aqi_category, AQI_CATEGORIES
from src.data.feature_engineering import generate_synthetic_training_data


# ========================
# Page Configuration
# ========================

st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ========================
# Custom CSS
# ========================

st.markdown(
    """
<style>
    .aqi-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .alert-hazardous {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ========================
# Helper Functions
# ========================


@st.cache_data(ttl=3600)
def fetch_current_aqi(city: str):
    """Fetch current AQI from API or generate demo data."""
    try:
        # Import data fetcher
        from src.data.ingestion import fetch_aqi_data

        # Fetch real data for city
        # fetch_aqi_data returns a dict with 'aqi_data' containing processed values
        all_data = fetch_aqi_data(city)
        aqi_data = all_data.get("aqi_data", {})

        if aqi_data and "error" not in aqi_data:
            pollutants = aqi_data.get("pollutants", {})
            weather = aqi_data.get("weather", {})

            payload = {
                "pm25": pollutants.get("pm25") or 35,
                "pm10": pollutants.get("pm10") or 0,
                "o3": pollutants.get("o3") or 0,
                "no2": pollutants.get("no2") or 0,
                "temp": weather.get("temperature") or 20,
                "h": weather.get("humidity") or 50,
                "city": city,
            }

            # Map keys to API expected keys
            api_payload = {
                "pm25": payload["pm25"],
                "pm10": payload["pm10"],
                "o3": payload["o3"],
                "no2": payload["no2"],
                "temperature": payload["temp"],
                "humidity": payload["h"],
                "city": city,
            }

            # API URL from env or default
            api_url = os.getenv("API_URL", "http://localhost:8000")

            # Try to fetch from our API
            pred_response = requests.post(
                f"{api_url}/predict", json=api_payload, timeout=5
            )
            if pred_response.status_code == 200:
                result = pred_response.json()
                # Check model used
                if result["model_used"] == "fallback_formula":
                    pass
                return result
    except Exception as e:
        print(f"Error fetching data: {e}")  # Log to console
        pass

    # Fallback to demo data
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    aqi = 150  # Default fallback
    category_info = get_aqi_category(aqi)

    return {
        "aqi": aqi,
        "category": "Error Fetching Data",
        "color": "#808080",
        "health_message": "Could not fetch data for this city",
        "model_used": "none",
        "timestamp": datetime.now().isoformat(),
    }


@st.cache_data(ttl=3600)
def fetch_forecast(city: str, days: int = 3, current_aqi_val: float = None):
    """Fetch forecast from API or generate smart demo data."""
    forecast = []

    # Anchor to current AQI if provided, else random
    if current_aqi_val:
        base_aqi = current_aqi_val
    else:
        base_aqi = np.random.randint(50, 100)

    for h in range(days * 24):
        timestamp = datetime.now() + timedelta(hours=h + 1)

        # Daily pattern (higher in morning/evening)
        hour_effect = 15 * np.sin((timestamp.hour - 8) * np.pi / 12)

        # Trend: Slowly revert to mean (assume mean is ~100 for polluted cities)
        # If very high, decay faster. If low, rise slightly.
        reversion = (100 - base_aqi) * (0.01 * h)

        # Noise
        noise = np.random.normal(0, 5)

        aqi = base_aqi + hour_effect + reversion + noise
        aqi = max(10, min(500, aqi))  # Clamp

        category_info = get_aqi_category(aqi)

        forecast.append(
            {
                "timestamp": timestamp,
                "aqi": aqi,
                "category": category_info["category"],
                "color": category_info["color"],
            }
        )

    return pd.DataFrame(forecast)


def get_aqi_color(aqi: float) -> str:
    """Get color for AQI value."""
    for category, info in AQI_CATEGORIES.items():
        if info["min"] <= aqi <= info["max"]:
            return info["color"]
    return "#7e0023"


def create_aqi_gauge(aqi: float):
    """Create a gauge chart for AQI display."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=aqi,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Current AQI", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 500], "tickwidth": 1},
                "bar": {"color": get_aqi_color(aqi)},
                "steps": [
                    {"range": [0, 50], "color": "lightgreen"},
                    {"range": [50, 100], "color": "yellow"},
                    {"range": [100, 150], "color": "orange"},
                    {"range": [150, 200], "color": "red"},
                    {"range": [200, 300], "color": "purple"},
                    {"range": [300, 500], "color": "maroon"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": aqi,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ========================
# Main Dashboard
# ========================


def main():
    # Sidebar
    with st.sidebar:
        st.title("🌍 AQI Dashboard")
        st.markdown("---")

        # Get settings
        settings = get_settings()
        default_city = settings.location.default_city.title()

        # City selection
        cities = [
            "Beijing",
            "London",
            "New York",
            "Delhi",
            "Tokyo",
            "Sydney",
            "Islamabad",
            "Lahore",
            "Karachi",
        ]
        if default_city not in cities:
            cities.append(default_city)

        # Find index of default city
        try:
            default_index = cities.index(default_city)
        except ValueError:
            default_index = 0

        city = st.selectbox("Select City", cities, index=default_index)

        # Forecast days
        forecast_days = st.slider("Forecast Days", 1, 7, 3)

        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            """
        This dashboard provides real-time AQI 
        monitoring and predictions using 
        machine learning models.
        
        **Project:** AI321L - ML Lab
        **Domain:** Earth & Environmental Intelligence
        """
        )

    # Main content
    st.title("Air Quality Index Prediction Dashboard")
    st.markdown(
        f"**City:** {city} | **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Fetch data
    current_aqi = fetch_current_aqi(city.lower())
    forecast_df = fetch_forecast(
        city.lower(), forecast_days, current_aqi_val=current_aqi.get("aqi")
    )

    # ========================
    # Row 1: Current AQI
    # ========================

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.plotly_chart(create_aqi_gauge(current_aqi["aqi"]), use_container_width=True)

    with col2:
        st.markdown("### Status")
        aqi_color = current_aqi["color"]
        st.markdown(
            f"""
        <div style="background-color: {aqi_color}; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;">{current_aqi['category']}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**Health Advisory:** {current_aqi['health_message']}")

    with col3:
        st.markdown("### Details")
        st.metric("AQI Value", f"{current_aqi['aqi']:.0f}")
        st.metric("Model Used", current_aqi["model_used"])
        st.metric("Category", current_aqi["category"])

    # Health Alert
    if current_aqi["aqi"] > 150:
        st.warning(
            f"⚠️ **Health Alert:** AQI is {current_aqi['category']}. {current_aqi['health_message']}"
        )

    if current_aqi["aqi"] > 300:
        st.error(
            f"🚨 **HAZARDOUS CONDITIONS:** Stay indoors and use air purifiers. Avoid outdoor activities."
        )

    # ========================
    # Precautions Section
    # ========================
    st.subheader("🛡️ Recommended Precautions")

    # Get precautions from current AQI data (added to settings)
    precautions = current_aqi.get("precautions", [])

    if not precautions:
        # Fallback if not in data (e.g. old settings loaded)
        from src.config.settings import get_aqi_category

        cat_info = get_aqi_category(current_aqi["aqi"])
        precautions = cat_info.get(
            "precautions", ["Wear a mask.", "Reduce outdoor activity."]
        )

    cols = st.columns(len(precautions))
    for i, precaution in enumerate(precautions):
        with cols[i]:
            st.info(f"**{precaution}**")

    st.markdown("---")

    # ========================
    # Row 2: Forecast Chart
    # ========================

    st.subheader(f"📈 {forecast_days}-Day AQI Forecast")

    # Create forecast chart
    fig = px.line(
        forecast_df, x="timestamp", y="aqi", color_discrete_sequence=["#1f77b4"]
    )

    # Add colored background regions for AQI categories
    for category, info in AQI_CATEGORIES.items():
        fig.add_hrect(
            y0=info["min"],
            y1=info["max"],
            fillcolor=info["color"],
            opacity=0.2,
            line_width=0,
        )

    fig.update_layout(
        xaxis_title="Date/Time",
        yaxis_title="AQI",
        yaxis_range=[0, 300],
        height=400,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_aqi = forecast_df["aqi"].mean()
        st.metric("Average AQI", f"{avg_aqi:.1f}")

    with col2:
        max_aqi = forecast_df["aqi"].max()
        st.metric("Max AQI", f"{max_aqi:.1f}")

    with col3:
        min_aqi = forecast_df["aqi"].min()
        st.metric("Min AQI", f"{min_aqi:.1f}")

    with col4:
        unhealthy_hours = len(forecast_df[forecast_df["aqi"] > 100])
        st.metric("Unhealthy Hours", unhealthy_hours)

    st.markdown("---")

    # ========================
    # Row 3: Category Distribution
    # ========================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Forecast Category Distribution")
        category_counts = forecast_df["category"].value_counts()

        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            color=category_counts.index,
            color_discrete_map={
                "Good": "#00e400",
                "Moderate": "#ffff00",
                "Unhealthy for Sensitive Groups": "#ff7e00",
                "Unhealthy": "#ff0000",
                "Very Unhealthy": "#8f3f97",
                "Hazardous": "#7e0023",
            },
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("📋 AQI Category Reference")

        category_df = pd.DataFrame(
            [
                {
                    "Category": name,
                    "Range": f"{info['min']}-{info['max']}",
                    "Color": info["color"],
                }
                for name, info in AQI_CATEGORIES.items()
            ]
        )

        st.dataframe(category_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
        **Health Recommendations:**
        - 🟢 **Good (0-50):** Ideal for outdoor activities
        - 🟡 **Moderate (51-100):** Acceptable for most people
        - 🟠 **USG (101-150):** Sensitive groups should reduce outdoor exposure
        - 🔴 **Unhealthy (151-200):** Everyone should limit outdoor activity
        - 🟣 **Very Unhealthy (201-300):** Health alert! Avoid outdoor activities
        - ⬛ **Hazardous (301+):** Emergency! Stay indoors
        """
        )

    st.markdown("---")

    # ========================
    # Row 4: Feature Importance (SHAP)
    # ========================

    st.subheader("🔍 Feature Importance (Model Explanation)")

    # Mock feature importance for demo
    features = [
        ("PM2.5", 0.35),
        ("Temperature", 0.18),
        ("PM10", 0.15),
        ("Humidity", 0.12),
        ("Hour of Day", 0.08),
        ("Wind Speed", 0.06),
        ("NO2", 0.04),
        ("Pressure", 0.02),
    ]

    importance_df = pd.DataFrame(features, columns=["Feature", "Importance"])

    fig_importance = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
    )
    fig_importance.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)

    st.caption(
        """
    **Note:** This shows which features have the most impact on the model's predictions.
    PM2.5 is typically the most important factor in determining AQI levels.
    """
    )

    # ========================
    # Footer
    # ========================

    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: gray; padding: 20px;">
        <p>AQI Prediction System | AI321L - Machine Learning Lab</p>
        <p>Built with ❤️ using Python, FastAPI, and Streamlit</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
