import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from pathlib import Path
from zoneinfo import ZoneInfo


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SafeAlert | Disaster Risk Intelligence",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def _streamlit_version():
    parts = []
    for piece in st.__version__.split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


# Newer Streamlit replaced use_container_width with width="stretch".
if _streamlit_version() >= (1, 50):
    STRETCH = {"width": "stretch"}
else:
    STRETCH = {"use_container_width": True}


# =====================================================
# CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, .stApp, button, input, select, textarea { font-family: 'Inter', sans-serif; }
    .stApp {
        background: radial-gradient(1100px 600px at 8% -10%, #16335f 0%, #0b1220 55%);
        color: #e5e7eb;
    }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1250px; }
    [data-testid="stSidebar"] { background: #0e1829; border-right: 1px solid #1f3150; }
    h1, h2, h3 { color: #e5efff; letter-spacing: -0.01em; }
    [data-testid="stMetric"] {
        background: linear-gradient(160deg, #15233d, #101b30);
        border: 1px solid #28406a;
        padding: 16px 18px;
        border-radius: 16px;
    }
    [data-testid="stMetricLabel"] { color: #9fb3cf; }
    [data-testid="stMetricValue"] { font-weight: 800; }
    div.stButton > button {
        border-radius: 10px; font-weight: 600; border: 0;
        background: linear-gradient(120deg, #2563eb, #7c3aed); color: #fff;
    }
    div.stButton > button:hover { filter: brightness(1.12); }
    .hero {
        padding: 26px 30px; border-radius: 22px; margin-bottom: 18px;
        background: linear-gradient(120deg, #0f4c75 0%, #1b2a6b 55%, #4a1d7a 100%);
        border: 1px solid #355a96;
        box-shadow: 0 10px 40px rgba(20, 60, 140, 0.35);
    }
    .hero h1 { margin: 0; font-size: 2.5rem; font-weight: 800; color: #fff; }
    .hero p { margin: 6px 0 12px; color: #c3d3ee; font-size: 1.02rem; }
    .badge {
        display: inline-block; padding: 4px 11px; border-radius: 999px; margin: 2px 6px 2px 0;
        background: rgba(255,255,255,0.12); color: #e3edff; font-size: 0.78rem;
    }
    .loc-strip {
        display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
        padding: 12px 16px; border-radius: 14px; margin: 10px 0 16px;
        background: #12203a; border: 1px solid #27406a; color: #e5efff;
    }
    .risk-card {
        padding: 18px 20px; border-radius: 18px; margin-bottom: 4px;
        background: linear-gradient(160deg, #15233d, #101b30); border: 2px solid;
    }
    .rc-title { color: #9fb3cf; font-size: 0.9rem; font-weight: 600; }
    .rc-label { font-size: 2.4rem; font-weight: 800; line-height: 1.1; }
    .rc-note { color: #9fb3cf; font-size: 0.8rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# =====================================================
# PROJECT PATHS AND DATASET LOADING
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

# The first file found in this list is used.
DATASET_CANDIDATES = [
    "landslide_clean.csv",
    "landslides_final.csv",
    "landslides_clean.csv",
    "landslides_real_sample.csv",
]


def find_dataset_path():
    for name in DATASET_CANDIDATES:
        candidate = BASE_DIR / name
        if candidate.exists():
            return candidate
    return None


DATASET_PATH = find_dataset_path()


@st.cache_data(show_spinner=False)
def load_dataset(file_path, modified_time):
    return pd.read_csv(file_path)


landslide_df = pd.DataFrame()
dataset_error = None

if DATASET_PATH is not None:
    try:
        landslide_df = load_dataset(
            str(DATASET_PATH),
            DATASET_PATH.stat().st_mtime
        )
    except Exception as error:
        dataset_error = str(error)
else:
    dataset_error = (
        "No dataset found. Put landslide_clean.csv in the same "
        "folder as app.py."
    )


# =====================================================
# COLUMN DETECTION
# =====================================================

def find_column(dataframe, possible_names):
    if dataframe.empty:
        return None

    normalized = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in dataframe.columns
    }

    for name in possible_names:
        if name in normalized:
            return normalized[name]

    return None


LAT_COL = find_column(landslide_df, ["latitude", "lat", "y"])
LON_COL = find_column(landslide_df, ["longitude", "lon", "lng", "long", "x"])
DISTRICT_COL = find_column(landslide_df, ["district", "district_name", "admin2"])
LOCATION_COL = find_column(landslide_df, ["location", "place", "locality", "site"])
STATE_COL = find_column(landslide_df, ["state", "state_name", "admin1"])
MATERIAL_COL = find_column(landslide_df, ["material", "material_involved"])
MOVEMENT_COL = find_column(landslide_df, ["movement_type", "movement", "type"])


# =====================================================
# HISTORICAL DATA HELPERS
# =====================================================

def coverage_text():
    """Describe which states the historical dataset covers."""
    if landslide_df.empty or STATE_COL is None:
        return "the regions in the historical dataset"

    states = sorted(
        landslide_df[STATE_COL].dropna().astype(str).unique()
    )

    return ", ".join(states) if states else "the dataset regions"


def nearby_records(latitude, longitude, radius_km):
    """Historical records within radius_km of a point (haversine)."""
    if landslide_df.empty or LAT_COL is None or LON_COL is None:
        return pd.DataFrame()

    lat = pd.to_numeric(landslide_df[LAT_COL], errors="coerce")
    lon = pd.to_numeric(landslide_df[LON_COL], errors="coerce")

    valid = lat.notna() & lon.notna()

    earth_radius = 6371.0

    phi1 = np.radians(latitude)
    phi2 = np.radians(lat[valid].to_numpy())
    d_phi = phi2 - phi1
    d_lambda = np.radians(lon[valid].to_numpy() - longitude)

    a = (
        np.sin(d_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    )

    distance = 2 * earth_radius * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )

    result = landslide_df.loc[valid].copy()
    result["distance_km"] = np.round(distance, 1)

    result = result[result["distance_km"] <= radius_km]

    return result.sort_values("distance_km")


def historical_figure(dataframe):
    hover_columns = [
        column
        for column in [STATE_COL, DISTRICT_COL, MATERIAL_COL, MOVEMENT_COL]
        if column is not None
    ]

    options = dict(
        lat=LAT_COL,
        lon=LON_COL,
        hover_name=LOCATION_COL,
        hover_data=hover_columns,
        zoom=4,
        height=500,
        title="Recorded Historical Landslide Locations"
    )

    if hasattr(px, "scatter_map"):
        return px.scatter_map(
            dataframe, map_style="open-street-map", **options
        )

    return px.scatter_mapbox(
        dataframe, mapbox_style="open-street-map", **options
    )


def render_historical_map():
    if landslide_df.empty or LAT_COL is None or LON_COL is None:
        st.info(
            "Historical incident mapping requires a dataset with "
            "latitude and longitude columns."
        )
        return

    historical_map = landslide_df.copy()

    historical_map[LAT_COL] = pd.to_numeric(
        historical_map[LAT_COL], errors="coerce"
    )
    historical_map[LON_COL] = pd.to_numeric(
        historical_map[LON_COL], errors="coerce"
    )

    historical_map = historical_map.dropna(subset=[LAT_COL, LON_COL])

    historical_map = historical_map[
        historical_map[LAT_COL].between(-90, 90)
        & historical_map[LON_COL].between(-180, 180)
    ]

    if historical_map.empty:
        st.info("No usable coordinates were found in the dataset.")
        return

    if STATE_COL is not None:
        state_options = ["All states"] + sorted(
            historical_map[STATE_COL].dropna().astype(str).unique()
        )

        chosen_state = st.selectbox(
            "Filter map by state", state_options, key="map_state_filter"
        )

        if chosen_state != "All states":
            historical_map = historical_map[
                historical_map[STATE_COL].astype(str) == chosen_state
            ]

    total_points = len(historical_map)
    max_points = 4000

    if total_points > max_points:
        historical_map = historical_map.sample(max_points, random_state=1)

    st.plotly_chart(historical_figure(historical_map), **STRETCH)

    note = f"Showing {len(historical_map):,} of {total_points:,} records"
    if total_points > max_points:
        note += " (random sample for speed; filter by state to see all)"

    st.caption(
        f"{note}. The dataset covers: {coverage_text()}. Points show "
        "where landslides were recorded, not current risk."
    )


def render_nearby(place, radius_km, show_table):
    near = nearby_records(
        place["latitude"], place["longitude"], radius_km
    )

    st.metric(
        f"Recorded landslides within {radius_km} km",
        f"{len(near):,}"
    )

    if near.empty:
        st.caption(
            f"No records in the dataset within {radius_km} km. "
            f"The dataset only covers: {coverage_text()}."
        )
        return

    if show_table:
        columns = [
            column
            for column in [
                LOCATION_COL, DISTRICT_COL, STATE_COL,
                MATERIAL_COL, MOVEMENT_COL, "distance_km"
            ]
            if column is not None
        ]

        st.dataframe(
            near[columns].head(10),
            hide_index=True,
            **STRETCH
        )

        st.caption(
            "Nearest historical records. This is context only and "
            "is not part of the risk score."
        )


# =====================================================
# WEATHER API
# =====================================================

@st.cache_data(ttl=1800)
def search_location(location):
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": location,
            "count": 10,
            "language": "en",
            "format": "json",
            "countryCode": "IN"
        },
        timeout=15
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    maharashtra = [
        result for result in results
        if result.get("admin1", "").lower() == "maharashtra"
    ]

    return maharashtra or results


@st.cache_data(ttl=1800)
def get_weather(latitude, longitude, timezone):
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "past_days": 3,
            "forecast_days": 4,
            "hourly": [
                "precipitation",
                "precipitation_probability",
                "soil_moisture_0_to_7cm",
                "relative_humidity_2m"
            ],
            "daily": [
                "precipitation_sum",
                "precipitation_probability_max",
                "temperature_2m_max",
                "temperature_2m_min"
            ]
        },
        timeout=25
    )

    response.raise_for_status()

    return response.json()


# =====================================================
# RISK ASSESSMENT ENGINE
# =====================================================

def risk_label(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def risk_color(label):
    return {
        "Low": "#22c55e",
        "Moderate": "#f59e0b",
        "High": "#ef4444"
    }.get(label, "#94a3b8")


def landslide_breakdown(rainfall_24h, rainfall_72h, soil_moisture):
    """Points contributed by each factor (max 50 / 35 / 15)."""
    p24 = 0
    if rainfall_24h >= 100:
        p24 = 50
    elif rainfall_24h >= 50:
        p24 = 35
    elif rainfall_24h >= 20:
        p24 = 20
    elif rainfall_24h >= 5:
        p24 = 8

    p72 = 0
    if rainfall_72h >= 200:
        p72 = 35
    elif rainfall_72h >= 100:
        p72 = 25
    elif rainfall_72h >= 50:
        p72 = 12

    psoil = 0
    if soil_moisture is not None:
        if soil_moisture >= 0.40:
            psoil = 15
        elif soil_moisture >= 0.30:
            psoil = 8

    return [
        ("Forecast rain, next 24h", p24, 50),
        ("Rain, previous 72h", p72, 35),
        ("Topsoil moisture", psoil, 15),
    ]


def calculate_landslide_risk(rainfall_24h, rainfall_72h, soil_moisture):
    """Experimental screening score, not a trained ML model."""
    parts = landslide_breakdown(rainfall_24h, rainfall_72h, soil_moisture)
    return min(sum(points for _, points, _ in parts), 100)


def risk_gauge(score, label, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 40, "color": "#e5efff"}},
        title={"text": title, "font": {"size": 16, "color": "#9fb3cf"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4b5d78"},
            "bar": {"color": risk_color(label), "thickness": 0.35},
            "bgcolor": "#111c2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#14301f"},
                {"range": [40, 70], "color": "#3a2f12"},
                {"range": [70, 100], "color": "#3b1717"},
            ],
        },
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=25, r=25, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5efff"},
    )
    return fig


def risk_card(title, label, note):
    color = risk_color(label)
    st.markdown(
        f'<div class="risk-card" style="border-color:{color}">'
        f'<div class="rc-title">{title}</div>'
        f'<div class="rc-label" style="color:{color}">{label}</div>'
        f'<div class="rc-note">{note}</div></div>',
        unsafe_allow_html=True
    )


def breakdown_chart(parts):
    frame = pd.DataFrame(parts, columns=["Factor", "Points", "Max"])
    frame["Label"] = frame.apply(lambda r: f"{r.Points} of {r.Max}", axis=1)

    fig = px.bar(
        frame, x="Points", y="Factor", orientation="h",
        text="Label", range_x=[0, 55]
    )
    fig.update_traces(marker_color="#38bdf8", textposition="outside")
    fig.update_layout(
        height=230, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", title=None),
        xaxis=dict(title="Score points contributed"),
    )
    return fig


def calculate_flood_risk(rainfall_24h, rainfall_72h, rain_probability):
    """Experimental rainfall-based flood screening score."""
    score = 0

    if rainfall_24h >= 100:
        score += 65
    elif rainfall_24h >= 50:
        score += 45
    elif rainfall_24h >= 20:
        score += 25
    elif rainfall_24h >= 5:
        score += 10

    if rainfall_72h >= 200:
        score += 25
    elif rainfall_72h >= 100:
        score += 15
    elif rainfall_72h >= 50:
        score += 8

    if rain_probability >= 80:
        score += 10
    elif rain_probability >= 60:
        score += 5

    return min(score, 100)


# =====================================================
# WEATHER DATA PROCESSING
# =====================================================

def prepare_weather_data(weather):
    hourly_df = pd.DataFrame(weather.get("hourly", {}))
    daily_df = pd.DataFrame(weather.get("daily", {}))

    if hourly_df.empty:
        raise ValueError("Hourly weather data is unavailable.")

    if "time" not in hourly_df.columns:
        raise ValueError("Weather timestamps are unavailable.")

    hourly_df["time"] = pd.to_datetime(hourly_df["time"], errors="coerce")
    hourly_df = hourly_df.dropna(subset=["time"])

    timezone_name = weather.get("timezone", "Asia/Kolkata")

    now_local = pd.Timestamp.now(
        tz=ZoneInfo(timezone_name)
    ).tz_localize(None)

    past_24 = hourly_df[
        (hourly_df["time"] <= now_local)
        & (hourly_df["time"] > now_local - pd.Timedelta(hours=24))
    ]

    past_72 = hourly_df[
        (hourly_df["time"] <= now_local)
        & (hourly_df["time"] > now_local - pd.Timedelta(hours=72))
    ]

    next_24 = hourly_df[
        (hourly_df["time"] >= now_local)
        & (hourly_df["time"] < now_local + pd.Timedelta(hours=24))
    ]

    if next_24.empty:
        raise ValueError("No upcoming hourly forecast is available.")

    def rainfall_sum(frame):
        if frame.empty or "precipitation" not in frame.columns:
            return 0.0

        return float(
            pd.to_numeric(
                frame["precipitation"], errors="coerce"
            ).fillna(0).sum()
        )

    rain_probability = 0.0

    if "precipitation_probability" in next_24.columns:
        probabilities = pd.to_numeric(
            next_24["precipitation_probability"], errors="coerce"
        ).dropna()

        if not probabilities.empty:
            rain_probability = float(probabilities.max())

    soil_moisture = None

    if "soil_moisture_0_to_7cm" in next_24.columns:
        soil_values = pd.to_numeric(
            next_24["soil_moisture_0_to_7cm"], errors="coerce"
        ).dropna()

        if not soil_values.empty:
            soil_moisture = float(soil_values.mean())

    return {
        "hourly_df": hourly_df,
        "daily_df": daily_df,
        "rainfall_past_24": rainfall_sum(past_24),
        "rainfall_past_72": rainfall_sum(past_72),
        "rainfall_next_24": rainfall_sum(next_24),
        "rain_probability": rain_probability,
        "soil_moisture": soil_moisture
    }


# =====================================================
# SIDEBAR
# =====================================================

PAGE_HOME = "🏠 Home"
PAGE_LANDSLIDE = "⛰️ Landslide Prediction"
PAGE_FLOOD = "🌊 Flood Risk"
PAGE_MAP = "🗺️ Risk Map"
PAGE_ANALYTICS = "📊 Analytics"
PAGE_HISTORY = "📁 Historical Records"
PAGE_ALERTS = "🚨 Alerts"
PAGE_ABOUT = "ℹ️ About"

st.sidebar.title("🌧️ SafeAlert")
st.sidebar.caption("Disaster Risk Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        PAGE_HOME, PAGE_LANDSLIDE, PAGE_FLOOD, PAGE_MAP,
        PAGE_ANALYTICS, PAGE_HISTORY, PAGE_ALERTS, PAGE_ABOUT
    ]
)

st.sidebar.divider()

radius_km = st.sidebar.slider(
    "Nearby records radius (km)",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

st.sidebar.markdown("**Primary objective:** Landslide prediction")

if not landslide_df.empty:
    st.sidebar.success(f"Dataset loaded: {len(landslide_df):,} records")
    st.sidebar.caption(f"File: {DATASET_PATH.name}")
else:
    st.sidebar.warning("Historical dataset unavailable.")


# =====================================================
# SHARED PAGE PIECES
# =====================================================

def render_header():
    st.markdown(
        '<div class="hero"><h1>🌧️ SafeAlert</h1>'
        '<p>Landslide risk screening and flood monitoring from live '
        'weather and real historical records.</p>'
        '<span class="badge">⛰️ Landslide-focused</span>'
        '<span class="badge">🌦️ Live Open-Meteo data</span>'
        '<span class="badge">📁 26,000+ real records</span>'
        '<span class="badge">🧪 Experimental screening</span></div>',
        unsafe_allow_html=True
    )


def render_footer():
    st.divider()
    st.caption(
        "SafeAlert | Landslide-focused disaster-risk prototype | "
        "Experimental screening, not an official warning system"
    )


def render_location_picker():
    if "selected_place" not in st.session_state:
        st.session_state.selected_place = None

    location_col, button_col = st.columns([4, 1])

    with location_col:
        location_query = st.text_input(
            "📍 Search a location",
            value="Pune",
            placeholder="Enter a city or location",
            key="location_input"
        )

    with button_col:
        st.write("")
        search_button = st.button("Search", type="primary", **STRETCH)

    if search_button:
        if not location_query.strip():
            st.warning("Please enter a location.")
        else:
            try:
                with st.spinner("Searching for the location..."):
                    results = search_location(location_query.strip())

                if results:
                    st.session_state.location_results = results
                    st.session_state.selected_place = None
                else:
                    st.warning("Location not found. Try another city.")

            except requests.RequestException:
                st.error(
                    "Location search failed. Check your internet "
                    "connection and try again."
                )

    results = st.session_state.get("location_results")

    if results:
        def format_location(index):
            item = results[index]
            parts = [
                item.get("name", "Unknown"),
                item.get("admin2", ""),
                item.get("admin1", "")
            ]
            return ", ".join(part for part in parts if part)

        chosen = st.selectbox(
            "Select the correct location",
            options=range(len(results)),
            format_func=format_location
        )

        if st.button("Use Selected Location", type="primary"):
            st.session_state.selected_place = results[chosen]


# =====================================================
# PAGES THAT DO NOT NEED A LOCATION
# =====================================================

def render_history_page():
    st.header("📁 Historical Landslide Records")

    if landslide_df.empty:
        st.error(dataset_error or "The dataset is empty.")
        st.write(f"Expected location: `{BASE_DIR}`")
        return

    st.success(f"Loaded {len(landslide_df):,} records from {DATASET_PATH.name}.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(landslide_df):,}")
    col2.metric("Total Columns", len(landslide_df.columns))
    col3.metric("Missing Cells", f"{int(landslide_df.isna().sum().sum()):,}")

    st.divider()
    st.subheader("Dataset Preview")

    search_text = st.text_input(
        "Search records", placeholder="Enter a keyword..."
    )

    display_df = landslide_df.copy()

    if search_text.strip():
        mask = display_df.astype(str).apply(
            lambda column: column.str.contains(
                search_text, case=False, na=False, regex=False
            )
        ).any(axis=1)

        display_df = display_df[mask]

    st.dataframe(display_df.head(500), hide_index=True, **STRETCH)

    st.caption(
        f"Showing up to 500 of {len(display_df):,} matching records."
    )

    st.divider()
    st.subheader("Dataset Column Information")

    info_df = pd.DataFrame({
        "Column": landslide_df.columns,
        "Data Type": [str(dtype) for dtype in landslide_df.dtypes],
        "Missing Values": [int(c) for c in landslide_df.isnull().sum()],
        "Unique Values": [
            int(landslide_df[column].nunique())
            for column in landslide_df.columns
        ]
    })

    st.dataframe(info_df, hide_index=True, **STRETCH)

    st.divider()
    st.subheader("Historical Locations")
    render_historical_map()

    st.divider()
    st.download_button(
        label="Download Historical CSV",
        data=landslide_df.to_csv(index=False).encode("utf-8"),
        file_name="landslide_clean.csv",
        mime="text/csv"
    )


def render_about_page():
    st.header("About SafeAlert")

    st.write(
        "SafeAlert is a student-developed disaster-risk monitoring "
        "prototype focused primarily on landslide risk assessment, "
        "with flood-risk monitoring as a secondary feature."
    )

    st.subheader("Current Features")
    st.markdown(
        """
        - Automatic location search and weather retrieval.
        - Recent and forecast rainfall summaries.
        - Experimental rainfall-based risk indicators.
        - Historical landslide records near a selected location.
        - Dataset exploration, analytics and map.
        - Safety guidance and official information links.
        """
    )

    st.subheader("Current Limitations")
    st.markdown(
        f"""
        - The scores are rule-based, not machine-learning predictions.
        - Rainfall alone cannot establish actual landslide risk.
        - Slope, geology, drainage and soil strength are not used yet.
        - The historical dataset only covers: {coverage_text()}.
        - Historical records are shown as context and are not part
          of the risk score.
        - The map does not display verified hazard boundaries.
        - Official warning feeds are not integrated.
        - Internet access is required for weather retrieval.
        """
    )

    st.subheader("Historical Dataset Status")

    if landslide_df.empty:
        st.warning("The historical dataset has not been loaded.")
    else:
        st.success(
            f"Loaded {len(landslide_df):,} records from {DATASET_PATH.name}."
        )
        st.code(", ".join(str(c) for c in landslide_df.columns))

    st.subheader("Weather Data Attribution")
    st.markdown("[Open-Meteo](https://open-meteo.com/)")


# =====================================================
# ROUTING
# =====================================================

render_header()

if page == PAGE_HISTORY:
    render_history_page()
    render_footer()
    st.stop()

if page == PAGE_ABOUT:
    render_about_page()
    render_footer()
    st.stop()

render_location_picker()

place = st.session_state.selected_place

if place is None:
    if page == PAGE_MAP:
        st.header("🗺️ Risk Monitoring Map")
        render_historical_map()
        st.info("Select a location above to also see its weather risk.")
    else:
        st.info(
            "Search for a location and select it to retrieve weather "
            "information and assess risk."
        )

        st.markdown("## Welcome to SafeAlert")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### ⛰️ Landslides")
            st.write(
                "Rainfall-related landslide indicators and nearby "
                "historical records."
            )

        with col2:
            st.markdown("### 🌦️ Weather")
            st.write("Forecasts retrieved automatically for a location.")

        with col3:
            st.markdown("### 📊 Historical Data")
            st.write("Analyze the real landslide records in the app.")

    render_footer()
    st.stop()


# =====================================================
# WEATHER RETRIEVAL AND RISK
# =====================================================

try:
    with st.spinner("Retrieving weather information..."):
        weather = get_weather(
            place["latitude"],
            place["longitude"],
            place.get("timezone", "Asia/Kolkata")
        )

        processed = prepare_weather_data(weather)

except requests.RequestException:
    st.error(
        "Unable to retrieve weather information. Check your internet "
        "connection and try again."
    )
    st.stop()

except (ValueError, KeyError, TypeError) as error:
    st.error(f"Unable to process weather information: {error}")
    st.stop()

hourly_df = processed["hourly_df"]
daily_df = processed["daily_df"]
rainfall_past_24 = processed["rainfall_past_24"]
rainfall_past_72 = processed["rainfall_past_72"]
rainfall_next_24 = processed["rainfall_next_24"]
rain_probability = processed["rain_probability"]
soil_moisture = processed["soil_moisture"]

landslide_score = calculate_landslide_risk(
    rainfall_next_24, rainfall_past_72, soil_moisture
)
flood_score = calculate_flood_risk(
    rainfall_next_24, rainfall_past_72, rain_probability
)

landslide_label = risk_label(landslide_score)
flood_label = risk_label(flood_score)

location_name = place.get("name", "Selected location")
district_name = place.get("admin2", "")
state_name = place.get("admin1", "")

elevation = weather.get("elevation")
elevation_text = f"{elevation:.0f} m" if elevation is not None else "n/a"
updated_text = pd.Timestamp.now(
    tz=ZoneInfo(weather.get("timezone", "Asia/Kolkata"))
).strftime("%d %b %Y, %H:%M")

place_line = ", ".join(
    part for part in [location_name, district_name, state_name] if part
)

st.markdown(
    f'<div class="loc-strip"><b>📍 {place_line}</b>'
    f'<span class="badge">{place["latitude"]:.4f}, {place["longitude"]:.4f}</span>'
    f'<span class="badge">⛰️ Elevation {elevation_text}</span>'
    f'<span class="badge">🕒 {updated_text}</span>'
    f'<span class="badge">Source: Open-Meteo</span></div>',
    unsafe_allow_html=True
)


# =====================================================
# HOME PAGE
# =====================================================

if page == PAGE_HOME:
    st.header("Risk overview")

    left, right = st.columns(2)

    with left:
        risk_card(
            "⛰️ LANDSLIDE RISK", landslide_label,
            "Based on rainfall and topsoil moisture"
        )
        st.plotly_chart(
            risk_gauge(landslide_score, landslide_label, "Landslide score"),
            **STRETCH
        )

    with right:
        risk_card(
            "🌊 FLOOD RISK", flood_label,
            "Based on rainfall and rain probability"
        )
        st.plotly_chart(
            risk_gauge(flood_score, flood_label, "Flood score"),
            **STRETCH
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rain, previous 24h", f"{rainfall_past_24:.1f} mm")
    c2.metric("Rain, previous 72h", f"{rainfall_past_72:.1f} mm")
    c3.metric("Forecast, next 24h", f"{rainfall_next_24:.1f} mm")
    c4.metric("Max rain probability", f"{rain_probability:.0f}%")

    st.divider()

    near_col, why_col = st.columns([1, 2])

    with near_col:
        st.subheader("History nearby")
        render_nearby(place, radius_km, show_table=False)

    with why_col:
        st.subheader("Why this landslide score?")
        st.plotly_chart(
            breakdown_chart(
                landslide_breakdown(
                    rainfall_next_24, rainfall_past_72, soil_moisture
                )
            ),
            **STRETCH
        )

    st.warning(
        "These are experimental screening indicators, not validated "
        "probabilities and not an official warning. No machine-learning "
        "model is used."
    )


# =====================================================
# LANDSLIDE PAGE
# =====================================================

elif page == PAGE_LANDSLIDE:
    st.header("⛰️ Landslide risk assessment")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rain, previous 24h", f"{rainfall_past_24:.1f} mm")
    c2.metric("Rain, previous 72h", f"{rainfall_past_72:.1f} mm")
    c3.metric("Forecast, next 24h", f"{rainfall_next_24:.1f} mm")
    c4.metric(
        "Topsoil moisture",
        f"{soil_moisture:.3f} m³/m³" if soil_moisture is not None else "n/a"
    )

    left, right = st.columns([1, 1])

    with left:
        risk_card(
            "LANDSLIDE RISK", landslide_label,
            f"Score {landslide_score}/100 (rule-based screening)"
        )
        st.plotly_chart(
            risk_gauge(landslide_score, landslide_label, "Landslide score"),
            **STRETCH
        )

    with right:
        st.subheader("What drives the score")
        st.plotly_chart(
            breakdown_chart(
                landslide_breakdown(
                    rainfall_next_24, rainfall_past_72, soil_moisture
                )
            ),
            **STRETCH
        )

        if landslide_label == "High":
            st.error(
                "Elevated score. Check official advisories, avoid "
                "unstable slopes and follow local instructions."
            )
        elif landslide_label == "Moderate":
            st.warning(
                "Moderate score. Monitor weather and official landslide "
                "advisories, especially near steep slopes."
            )
        else:
            st.success(
                "Low score. Landslides can still occur when rainfall "
                "indicators are low."
            )

    st.divider()
    st.subheader("Rainfall, past 3 days and next 3 days")

    if "precipitation" in hourly_df.columns:
        rain_fig = px.area(
            hourly_df, x="time", y="precipitation",
            labels={"time": "Time", "precipitation": "Rain (mm/h)"}
        )
        rain_fig.update_traces(line_color="#38bdf8", fillcolor="rgba(56,189,248,0.25)")
        rain_fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(rain_fig, **STRETCH)

    st.divider()
    st.subheader(f"Recorded landslides within {radius_km} km")
    render_nearby(place, radius_km, show_table=True)

    st.caption(
        "Rainfall-based screening only. It does not account for slope, "
        "geology, soil strength, drainage or verified local "
        "susceptibility, and is not a validated prediction model."
    )


# =====================================================
# FLOOD PAGE
# =====================================================

elif page == PAGE_FLOOD:
    st.header("🌊 Flood Risk Monitoring")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rainfall — Previous 24h", f"{rainfall_past_24:.1f} mm")
    col2.metric("Rainfall — Next 24h", f"{rainfall_next_24:.1f} mm")
    col3.metric("Rain Probability", f"{rain_probability:.0f}%")

    st.divider()
    st.subheader("Flood Screening Indicator")

    st.markdown(
        f"<h1 style='color:{risk_color(flood_label)}'>{flood_label}</h1>",
        unsafe_allow_html=True
    )

    st.progress(flood_score / 100)
    st.write(f"Experimental screening score: {flood_score}/100")

    if flood_label == "High":
        st.error(
            "Elevated rainfall-based screening score. Check official "
            "flood warnings and avoid floodwater and flooded roads."
        )
    elif flood_label == "Moderate":
        st.warning(
            "Monitor rainfall, local water levels and official "
            "weather advisories."
        )
    else:
        st.info(
            "The rainfall-based screening score is low. Flooding is "
            "still possible in conditions not represented here."
        )

    st.caption(
        "This assessment does not model river levels, drainage, "
        "catchment hydrology or verified flood boundaries."
    )


# =====================================================
# RISK MAP PAGE
# =====================================================

elif page == PAGE_MAP:
    st.header("🗺️ Risk Monitoring Map")

    st.map(
        pd.DataFrame({
            "latitude": [place["latitude"]],
            "longitude": [place["longitude"]]
        }),
        zoom=8
    )

    st.subheader("Selected Location")
    st.write(f"**Location:** {location_name}")
    st.write(f"**State:** {state_name}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Landslide Indicator", landslide_label)
    col2.metric("Flood Indicator", flood_label)

    with col3:
        render_nearby(place, radius_km, show_table=False)

    st.divider()
    st.subheader("Historical Landslide Locations")
    render_historical_map()

    st.caption(
        "The selected-location map is not a verified hazard map. "
        "Historical incident points do not establish current risk."
    )


# =====================================================
# ANALYTICS PAGE
# =====================================================

elif page == PAGE_ANALYTICS:
    st.header("📊 Weather & Historical Analytics")

    st.subheader("Rainfall Forecast")

    if not daily_df.empty and "precipitation_sum" in daily_df.columns:
        st.plotly_chart(
            px.bar(
                daily_df,
                x="time",
                y="precipitation_sum",
                title="Daily Precipitation",
                labels={"time": "Date", "precipitation_sum": "Rainfall (mm)"}
            ),
            **STRETCH
        )

    st.subheader("Hourly Rainfall")

    if "precipitation" in hourly_df.columns:
        st.plotly_chart(
            px.line(
                hourly_df,
                x="time",
                y="precipitation",
                title="Recent and Forecast Hourly Rainfall",
                labels={"time": "Time", "precipitation": "Rainfall (mm)"}
            ),
            **STRETCH
        )

    if (
        not daily_df.empty
        and {"temperature_2m_min", "temperature_2m_max"}.issubset(
            daily_df.columns
        )
    ):
        st.subheader("Temperature Forecast")

        st.plotly_chart(
            px.line(
                daily_df,
                x="time",
                y=["temperature_2m_min", "temperature_2m_max"],
                markers=True,
                labels={
                    "time": "Date",
                    "value": "Temperature (°C)",
                    "variable": "Measurement"
                }
            ),
            **STRETCH
        )

    if not landslide_df.empty:
        st.divider()
        st.subheader("Historical Dataset Summary")

        col1, col2 = st.columns(2)
        col1.metric("Historical Records", f"{len(landslide_df):,}")
        col2.metric("Dataset Columns", len(landslide_df.columns))

        for column, title in [
            (STATE_COL, "Records by State"),
            (DISTRICT_COL, "Records by District (top 15)"),
            (MATERIAL_COL, "Records by Material"),
        ]:
            if column is None:
                continue

            counts = (
                landslide_df[column]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .head(15)
                .rename_axis("Category")
                .reset_index(name="Records")
            )

            st.plotly_chart(
                px.bar(counts, x="Category", y="Records", title=title),
                **STRETCH
            )


# =====================================================
# ALERTS PAGE
# =====================================================

elif page == PAGE_ALERTS:
    st.header("🚨 Risk Monitoring & Safety Guidance")

    st.subheader("Landslide Monitoring")

    if landslide_label == "High":
        st.error(
            "Elevated rainfall-based landslide screening score. "
            "Consult official warnings, avoid unstable slopes and "
            "follow local emergency instructions."
        )
    elif landslide_label == "Moderate":
        st.warning(
            "Monitor official landslide and heavy-rainfall advisories, "
            "especially in hilly areas."
        )
    else:
        st.info(
            "The current screening score is low. Continue following "
            "official weather and safety advisories."
        )

    st.divider()
    st.subheader("Flood Monitoring")

    if flood_label == "High":
        st.error(
            "Elevated rainfall-based flood screening score. Check "
            "official warnings and avoid flooded roads."
        )
    elif flood_label == "Moderate":
        st.warning("Monitor weather forecasts and local conditions.")
    else:
        st.info("The current rainfall-based flood screening score is low.")

    st.divider()
    st.subheader("Official Information Sources")

    st.markdown(
        """
        - [India Meteorological Department](https://mausam.imd.gov.in/)
        - [Maharashtra State Disaster Management Authority](https://sdma.maharashtra.gov.in/)
        - [Open-Meteo Weather API](https://open-meteo.com/)
        """
    )

    st.caption(
        "SafeAlert does not send verified government alerts or "
        "automatically notify emergency services."
    )


render_footer()