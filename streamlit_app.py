from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import json
import uuid
import os

import folium
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_folium import st_folium


# =====================================================
# 1. PAGE CONFIGURATION AND PROJECT PATHS
# =====================================================

st.set_page_config(
    page_title="UrbanPulse AI",
    page_icon="🌆",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
#API_URL = "http://127.0.0.1:8000"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


ROAD_GEOJSON_PATH = (
    PROJECT_ROOT / "data" / "maps" / "hyderabad_road_segments.geojson"
)
TRAFFIC_CSV_PATH = (
    PROJECT_ROOT / "data" / "traffic" / "synthetic_traffic_records.csv"
)
WEATHER_JSON_PATH = (
    PROJECT_ROOT / "data" / "Weather" / "hyderabad_weather.json"
)

ADVISORY_JSON_PATH = (
    PROJECT_ROOT / "data" / "Advisories" / "durga_matha_immersion_2026.json"
)
ADVISORY_IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "Advisories"
    / "images"
    / "durga_matha_immersion_2026.jpeg"
)
ADVISORY_OCR_PATH = (
    PROJECT_ROOT / "data" / "processed" / "advisory_image_ocr.json"
)
ADVISORY_VLM_PATH = (
    PROJECT_ROOT / "data" / "processed" / "advisory_image_vlm.json"
)


# =====================================================
# 2. SESSION STATE
# =====================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# 3. CACHED DATA LOADING FUNCTIONS
# =====================================================

@st.cache_data(show_spinner="Loading the cleaned road network...")
def load_road_geojson(file_path: str):
    """Load and cache the cleaned road GeoJSON file."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Road GeoJSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("type") != "FeatureCollection":
        raise ValueError("The road GeoJSON must be a FeatureCollection.")

    features = data.get("features", [])
    if not features:
        raise ValueError("The road GeoJSON contains no features.")

    return data


@st.cache_data(show_spinner="Loading traffic simulation records...")
def load_traffic_csv(file_path: str) -> pd.DataFrame:
    """Load the traffic CSV and normalize key field types."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Traffic CSV file not found: {path}")

    df = pd.read_csv(path)

    required_columns = {
        "record_id",
        "simulation_time",
        "location",
        "road_type",
        "speed_kmh",
        "vehicles_per_minute",
        "congestion_level",
    }
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            "Traffic CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["simulation_time"] = pd.to_datetime(
        df["simulation_time"], errors="coerce"
    )
    df["speed_kmh"] = pd.to_numeric(df["speed_kmh"], errors="coerce")
    df["vehicles_per_minute"] = pd.to_numeric(
        df["vehicles_per_minute"], errors="coerce"
    )

    df["location"] = df["location"].fillna("Unknown").astype(str)
    df["road_type"] = df["road_type"].fillna("unknown").astype(str)
    df["congestion_level"] = (
        df["congestion_level"].fillna("unknown").astype(str).str.lower()
    )

    return df


@st.cache_data(show_spinner=False)
def load_json_file(file_path: str):
    """Load a JSON file, including files saved with a UTF-8 BOM."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


# =====================================================
# 4. ROAD NETWORK HELPER FUNCTIONS
# =====================================================

def get_road_type(feature: dict) -> str:
    """Read the road category while tolerating common property names."""
    properties = feature.get("properties") or {}
    value = (
        properties.get("highway")
        or properties.get("road_type")
        or properties.get("highway_type")
        or properties.get("class")
        or "unknown"
    )

    if isinstance(value, list):
        value = value[0] if value else "unknown"

    return str(value).strip().lower() or "unknown"


def get_map_center(features: list) -> tuple[float, float]:
    """Calculate a map centre from the coordinates in the GeoJSON."""
    min_lon = min_lat = float("inf")
    max_lon = max_lat = float("-inf")
    valid_points = 0

    def visit_coordinates(value):
        nonlocal min_lon, min_lat, max_lon, max_lat, valid_points

        if not isinstance(value, (list, tuple)):
            return

        # GeoJSON coordinate pairs are [longitude, latitude].
        if (
            len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            lon, lat = float(value[0]), float(value[1])
            if -180 <= lon <= 180 and -90 <= lat <= 90:
                min_lon = min(min_lon, lon)
                max_lon = max(max_lon, lon)
                min_lat = min(min_lat, lat)
                max_lat = max(max_lat, lat)
                valid_points += 1
            return

        for item in value:
            visit_coordinates(item)

    for feature in features:
        geometry = feature.get("geometry") or {}
        visit_coordinates(geometry.get("coordinates", []))

    if not valid_points:
        return 17.4401, 78.3489

    return (min_lat + max_lat) / 2, (min_lon + max_lon) / 2


def road_style(feature: dict) -> dict:
    """Assign a distinct visual style to each road category."""
    road_type = get_road_type(feature)

    colors = {
        "motorway": "#7B241C",
        "trunk": "#922B21",
        "primary": "#E74C3C",
        "secondary": "#E67E22",
        "tertiary": "#D4AC0D",
        "residential": "#2874A6",
        "service": "#7F8C8D",
        "unclassified": "#566573",
        "living_street": "#8E44AD",
        "road": "#34495E",
    }

    color = colors.get(road_type, "#95A5A6")
    weight = 4 if road_type in {"primary", "trunk", "motorway"} else 2

    return {
        "color": color,
        "weight": weight,
        "opacity": 0.85,
    }


# =====================================================
# 5. APPLICATION HEADER AND STYLING
# =====================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #1677C8;
    }
    .subtitle {
        font-size: 16px;
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🌆 UrbanPulse AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "Multimodal RAG-Based Urban Traffic Intelligence Platform"
    "</div>",
    unsafe_allow_html=True,
)
st.divider()


# =====================================================
# 6. SIDEBAR: BACKEND STATUS AND CHAT CONTROLS
# =====================================================

with st.sidebar:
    st.title("UrbanPulse AI")
    st.caption("City Intelligence Assistant")
    st.divider()

    st.subheader("Backend Status")
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=3)
        health_response.raise_for_status()
        health_data = health_response.json()

        if health_data.get("agent_ready"):
            st.success("AI Backend Connected")
        else:
            st.warning("AI Backend is not ready")
    except requests.exceptions.RequestException:
        st.error("FastAPI server is unavailable")

    st.divider()
    st.subheader("Capabilities")
    st.markdown(
        """
        - Traffic advisory retrieval
        - Road diversion information
        - Traffic congestion analysis
        - Weather information
        - Nearby road discovery
        - Conversational AI
        - Interactive road network map
        - Traffic analytics
        - Advisory image intelligence
        """
    )
    st.divider()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption("UrbanPulse AI | Development Version")


# =====================================================
# 7. MAIN NAVIGATION TABS
# =====================================================

chat_tab, map_tab, analytics_tab, advisory_tab, weather_tab = st.tabs(
    [
        "💬 AI Assistant",
        "🗺️ Road Network Map",
        "📊 Traffic Analytics",
        "📰 Advisory Intelligence",
        "🌤️ Weather Intelligence",
    ]
)


# =====================================================
# 8. AI ASSISTANT TAB
# =====================================================

with chat_tab:
    if not st.session_state.messages:
        st.subheader("Welcome to UrbanPulse AI!")
        st.write(
            "Explore traffic information, road advisories, diversions, "
            "weather snapshots and nearby roads using our AI-powered "
            "city intelligence assistant."
        )
        st.info("Ask a question below to start your conversation.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask UrbanPulse AI about traffic, roads or weather...",
        key="urbanpulse_chat_input",
    )

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("UrbanPulse AI is thinking..."):
                try:
                    payload = {
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                    }
                    response = requests.post(
                        f"{API_URL}/chat",
                        json=payload,
                        timeout=180,
                    )
                    response.raise_for_status()
                    result = response.json()
                    answer = result.get(
                        "answer", "No answer was generated."
                    )

                except requests.exceptions.Timeout:
                    answer = "The request took too long. Please try again."
                except requests.exceptions.ConnectionError:
                    answer = (
                        "Unable to connect to the UrbanPulse AI backend. "
                        "Please check whether FastAPI is running."
                    )
                except requests.exceptions.HTTPError as error:
                    answer = f"The API returned an error: {error}"
                except requests.exceptions.RequestException as error:
                    answer = f"API request failed: {error}"
                except ValueError:
                    answer = "The backend returned an invalid response."

            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


# =====================================================
# 9. ROAD NETWORK MAP TAB
# =====================================================

with map_tab:
    st.subheader("Hyderabad Road Network Explorer")
    st.caption(
        "Interactive visualization of the cleaned road-segment GeoJSON. "
        "The current extract covers the Gachibowli-area study region, "
        "not all of Hyderabad."
    )

    try:
        road_data = load_road_geojson(str(ROAD_GEOJSON_PATH))
        all_features = road_data["features"]

        road_types = sorted(
            {get_road_type(feature) for feature in all_features}
        )
        road_counts = {
            road_type: sum(
                1 for feature in all_features
                if get_road_type(feature) == road_type
            )
            for road_type in road_types
        }

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total road segments", f"{len(all_features):,}")
        metric_col2.metric("Road categories", len(road_types))
        metric_col3.metric("Study area", "Gachibowli region")

        selected_type = st.selectbox(
            "Filter by road category",
            options=["All road categories"] + road_types,
            help="Choose a road category to display only that type.",
        )

        if selected_type == "All road categories":
            visible_features = all_features
        else:
            visible_features = [
                feature for feature in all_features
                if get_road_type(feature) == selected_type
            ]

        st.caption(
            f"Displaying {len(visible_features):,} of "
            f"{len(all_features):,} road segments."
        )

        map_center = get_map_center(all_features)
        road_map = folium.Map(
            location=map_center,
            zoom_start=13,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        # Use only property fields common to all visible features.
        tooltip_fields = []
        candidate_fields = [
            "name", "highway", "road_type", "maxspeed",
            "lanes", "oneway", "surface", "source",
        ]

        if visible_features:
            common_property_keys = set(
                (visible_features[0].get("properties") or {}).keys()
            )

            for feature in visible_features[1:]:
                feature_keys = set(
                    (feature.get("properties") or {}).keys()
                )
                common_property_keys.intersection_update(feature_keys)

            tooltip_fields = [
                field for field in candidate_fields
                if field in common_property_keys
            ]

        tooltip = None
        if tooltip_fields:
            aliases = {
                "name": "Road name",
                "highway": "Road category",
                "road_type": "Road category",
                "maxspeed": "Max speed",
                "lanes": "Lanes",
                "oneway": "One-way",
                "surface": "Surface",
                "source": "Data source",
            }
            tooltip = folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=[
                    aliases.get(field, field.title())
                    for field in tooltip_fields
                ],
                sticky=True,
                localize=True,
            )

        folium.GeoJson(
            {"type": "FeatureCollection", "features": visible_features},
            name="Road segments",
            style_function=road_style,
            tooltip=tooltip,
        ).add_to(road_map)

        folium.LayerControl(collapsed=True).add_to(road_map)

        with st.expander("Plot a coordinate and radius on the map"):
            st.caption(
                "This control visualizes a point and radius only. "
                "It does not execute a nearby-road search."
            )
            show_search_area = st.checkbox(
                "Show coordinate marker and radius"
            )
            if show_search_area:
                coord_col1, coord_col2, coord_col3 = st.columns(3)

                with coord_col1:
                    latitude = st.number_input(
                        "Latitude",
                        min_value=-90.0,
                        max_value=90.0,
                        value=float(map_center[0]),
                        format="%.6f",
                    )

                with coord_col2:
                    longitude = st.number_input(
                        "Longitude",
                        min_value=-180.0,
                        max_value=180.0,
                        value=float(map_center[1]),
                        format="%.6f",
                    )

                with coord_col3:
                    radius_m = st.slider(
                        "Radius (metres)",
                        min_value=100,
                        max_value=2000,
                        value=500,
                        step=100,
                    )

                folium.Marker(
                    [latitude, longitude],
                    tooltip="Selected coordinate",
                    popup=(
                        f"Latitude: {latitude:.6f}<br>"
                        f"Longitude: {longitude:.6f}"
                    ),
                    icon=folium.Icon(color="red", icon="map-marker"),
                ).add_to(road_map)

                folium.Circle(
                    [latitude, longitude],
                    radius=radius_m,
                    color="#2874A6",
                    fill=True,
                    fill_opacity=0.12,
                    tooltip=f"Visual radius: {radius_m} m",
                ).add_to(road_map)

        st_folium(
            road_map,
            height=650,
            use_container_width=True,
            key="urbanpulse_road_network_map",
        )

        with st.expander("Road category counts"):
            count_rows = [
                {
                    "Road category": road_type,
                    "Segments": road_counts[road_type],
                }
                for road_type in road_types
            ]
            st.dataframe(
                count_rows,
                use_container_width=True,
                hide_index=True,
            )

    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "Check that data/maps/hyderabad_road_segments.geojson "
            "exists inside your UrbanPulseAI project folder."
        )
    except (json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not read the road GeoJSON: {error}")
    except Exception as error:
        st.error(f"The road map could not be rendered: {error}")
        st.caption("Check that folium and streamlit-folium are installed.")


# =====================================================
# 10. TRAFFIC ANALYTICS TAB
# =====================================================

with analytics_tab:
    st.subheader("Traffic Analytics Dashboard")
    st.caption(
        "Analysis of the project's generated traffic simulation records."
    )
    st.warning(
        "These records are synthetic simulations, not live traffic "
        "observations. Do not interpret these charts as current road conditions."
    )

    try:
        traffic_df = load_traffic_csv(str(TRAFFIC_CSV_PATH))
        traffic_df = traffic_df.dropna(
            subset=["simulation_time", "speed_kmh", "vehicles_per_minute"]
        )

        if traffic_df.empty:
            st.info("No valid traffic records are available for plotting.")
        else:
            total_records = len(traffic_df)
            mean_speed = traffic_df["speed_kmh"].mean()
            mean_flow = traffic_df["vehicles_per_minute"].mean()
            heavy_records = int(
                (traffic_df["congestion_level"] == "heavy").sum()
            )

            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("Simulation records", f"{total_records:,}")
            metric2.metric("Mean speed", f"{mean_speed:.1f} km/h")
            metric3.metric("Mean vehicle flow", f"{mean_flow:.1f}/min")
            metric4.metric("Heavy records", f"{heavy_records:,}")

            st.divider()

            # Chart 1: congestion distribution
            congestion_counts = (
                traffic_df["congestion_level"]
                .value_counts()
                .rename_axis("Congestion level")
                .reset_index(name="Record count")
            )

            # Chart 2: average speed by road type
            speed_by_road = (
                traffic_df.groupby("road_type", as_index=False)["speed_kmh"]
                .mean()
                .sort_values("speed_kmh", ascending=True)
            )

            # Chart 3: mean vehicles per minute by location
            flow_by_location = (
                traffic_df.groupby("location", as_index=False)[
                    "vehicles_per_minute"
                ]
                .mean()
                .sort_values("vehicles_per_minute", ascending=True)
            )

            # Chart 4: average simulated speed over six-hour intervals
            trend_df = traffic_df.copy()
            trend_df["time_bucket"] = (
                trend_df["simulation_time"].dt.floor("6h")
            )
            speed_trend = (
                trend_df.groupby("time_bucket", as_index=False)["speed_kmh"]
                .mean()
                .sort_values("time_bucket")
            )

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("#### Congestion distribution")
                fig_congestion = px.pie(
                    congestion_counts,
                    names="Congestion level",
                    values="Record count",
                    hole=0.45,
                )
                fig_congestion.update_layout(
                    legend_title_text="Congestion level",
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(
                    fig_congestion,
                    use_container_width=True,
                )

            with chart_col2:
                st.markdown("#### Average speed by road type")
                fig_speed = px.bar(
                    speed_by_road,
                    x="speed_kmh",
                    y="road_type",
                    orientation="h",
                    labels={
                        "speed_kmh": "Average speed (km/h)",
                        "road_type": "Road type",
                    },
                )
                fig_speed.update_layout(
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(
                    fig_speed,
                    use_container_width=True,
                )

            chart_col3, chart_col4 = st.columns(2)

            with chart_col3:
                st.markdown("#### Mean vehicle flow by location")
                fig_flow = px.bar(
                    flow_by_location,
                    x="vehicles_per_minute",
                    y="location",
                    orientation="h",
                    labels={
                        "vehicles_per_minute": "Mean vehicles per minute",
                        "location": "Location",
                    },
                )
                fig_flow.update_layout(
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(
                    fig_flow,
                    use_container_width=True,
                )

            with chart_col4:
                st.markdown("#### Simulated speed trend")
                fig_trend = px.line(
                    speed_trend,
                    x="time_bucket",
                    y="speed_kmh",
                    markers=True,
                    labels={
                        "time_bucket": "Simulation time",
                        "speed_kmh": "Average speed (km/h)",
                    },
                )
                fig_trend.update_layout(
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(
                    fig_trend,
                    use_container_width=True,
                )

            st.divider()
            st.markdown("#### Inspect traffic simulation records")

            locations = sorted(traffic_df["location"].unique().tolist())
            selected_location = st.selectbox(
                "Filter records by location",
                ["All locations"] + locations,
            )

            filtered_df = traffic_df
            if selected_location != "All locations":
                filtered_df = traffic_df[
                    traffic_df["location"] == selected_location
                ]

            display_columns = [
                column for column in [
                    "record_id",
                    "simulation_time",
                    "location",
                    "road_type",
                    "speed_kmh",
                    "vehicles_per_minute",
                    "congestion_level",
                    "record_origin",
                    "is_real_observation",
                ]
                if column in filtered_df.columns
            ]

            st.caption(
                f"Showing {len(filtered_df):,} of "
                f"{len(traffic_df):,} simulation records."
            )
            st.dataframe(
                filtered_df[display_columns].sort_values(
                    "simulation_time",
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )

    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "Check that data/traffic/synthetic_traffic_records.csv "
            "exists inside your UrbanPulseAI project folder."
        )
    except (pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not load traffic analytics data: {error}")


# =====================================================
# 11. ADVISORY INTELLIGENCE TAB
# =====================================================

with advisory_tab:
    st.subheader("Traffic Advisory Intelligence")
    st.caption(
        "Explore the original Hyderabad Traffic Police advisory, "
        "its structured diversion records, OCR extraction and VLM analysis."
    )

    try:
        advisory = load_json_file(str(ADVISORY_JSON_PATH))

        notice_period = advisory.get("published_notice_period", {})
        valid_from_text = notice_period.get("start")
        valid_until_text = notice_period.get("end")

        calculated_status = "unknown"
        valid_from = None
        valid_until = None

        if valid_from_text and valid_until_text:
            valid_from = datetime.fromisoformat(valid_from_text)
            valid_until = datetime.fromisoformat(valid_until_text)
            current_time = datetime.now(valid_until.tzinfo)

            if current_time < valid_from:
                calculated_status = "upcoming"
            elif current_time <= valid_until:
                calculated_status = "active"
            else:
                calculated_status = "historical_expired"

        title_col, status_col = st.columns([3, 1])

        with title_col:
            st.markdown(
                f"### {advisory.get('title', 'Traffic Advisory')}"
            )
            st.caption(
                f"Issued by: {advisory.get('source', 'Not specified')} "
                f"| Advisory ID: {advisory.get('advisory_id', 'Not specified')}"
            )

        with status_col:
            if calculated_status == "active":
                st.success("ACTIVE ADVISORY")
            elif calculated_status == "historical_expired":
                st.warning("HISTORICAL — EXPIRED")
            elif calculated_status == "upcoming":
                st.info("UPCOMING ADVISORY")
            else:
                st.warning("VALIDITY UNKNOWN")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric(
            "Junctions listed",
            len(advisory.get("junctions_to_avoid", [])),
        )
        metric2.metric(
            "Diversion instructions",
            len(advisory.get("diversions", [])),
        )
        metric3.metric(
            "Affected locations",
            len(advisory.get("affected_locations", [])),
        )

        if valid_from and valid_until:
            st.markdown("#### Advisory validity")
            date_col1, date_col2 = st.columns(2)
            date_col1.write(
                f"**Valid from:** {valid_from.isoformat()}"
            )
            date_col2.write(
                f"**Valid until:** {valid_until.isoformat()}"
            )

        if calculated_status == "historical_expired":
            st.warning(
                "This notice has expired. Its junction and diversion "
                "instructions are historical and must not be presented "
                "as current traffic restrictions."
            )
        elif calculated_status == "active":
            st.info(
                "The notice is within its stated validity period. "
                "This does not independently confirm live road conditions."
            )

        st.divider()

        # Original source image
        st.markdown("#### Original advisory image")
        if ADVISORY_IMAGE_PATH.exists():
            st.image(
                str(ADVISORY_IMAGE_PATH),
                caption="Original Hyderabad Traffic Police advisory image",
                use_container_width=True,
            )
        else:
            st.warning(
                "Original advisory image not found. Expected location: "
                f"{ADVISORY_IMAGE_PATH}"
            )

        source_url = advisory.get("source_url")
        if source_url:
            st.markdown(f"[Open original source notice]({source_url})")

        st.divider()

        # Junction list
        st.markdown("#### Junctions listed in this advisory")
        st.caption(
            "These are the junctions named by this specific notice; "
            "they are not a list of currently restricted junctions "
            "when the advisory has expired."
        )

        junctions = advisory.get("junctions_to_avoid", [])
        if junctions:
            junction_cols = st.columns(3)
            for index, junction in enumerate(junctions):
                with junction_cols[index % 3]:
                    st.markdown(f"- {junction}")
        else:
            st.info("No junction list is available in the advisory JSON.")

        # Diversion instructions
        st.markdown("#### Diversion instructions")
        diversions = advisory.get("diversions", [])

        if diversions:
            for index, diversion in enumerate(diversions, start=1):
                route_title = (
                    f"{index}. {diversion.get('from', 'Origin not specified')} "
                    f"→ {diversion.get('towards', 'Destination not specified')}"
                )

                with st.expander(route_title):
                    if diversion.get("diversion_at"):
                        st.write(
                            f"**Diversion point:** {diversion['diversion_at']}"
                        )

                    if diversion.get("restriction"):
                        st.write(
                            f"**Restriction:** {diversion['restriction']}"
                        )

                    alternatives = diversion.get("alternative_route", [])
                    if alternatives:
                        st.write(
                            "**Alternative route:** "
                            + " → ".join(alternatives)
                        )

                    if diversion.get("exception"):
                        st.write(
                            f"**Exception:** {diversion['exception']}"
                        )
        else:
            st.info(
                "No diversion instructions are available in the advisory JSON."
            )

        with st.expander("Affected locations and source metadata"):
            st.write("**Affected locations**")
            st.write(advisory.get("affected_locations", []))
            st.write("**Collection method**")
            st.write(advisory.get("collection_method", "Not specified"))
            st.write("**Verified from full notice**")
            st.write(
                advisory.get("verified_from_full_notice", "Not specified")
            )

        # OCR output
        st.divider()
        st.markdown("#### OCR text extraction")

        if ADVISORY_OCR_PATH.exists():
            try:
                ocr_data = load_json_file(str(ADVISORY_OCR_PATH))
                with st.expander("View OCR extraction output"):
                    st.json(ocr_data)
            except (json.JSONDecodeError, OSError, ValueError) as error:
                st.error(f"Could not read OCR output: {error}")
        else:
            st.info(
                "OCR output file not found. Expected: "
                f"{ADVISORY_OCR_PATH}"
            )

        # VLM output
        st.markdown("#### VLM visual analysis")

        if ADVISORY_VLM_PATH.exists():
            try:
                vlm_data = load_json_file(str(ADVISORY_VLM_PATH))
                with st.expander("View VLM analysis output"):
                    st.json(vlm_data)
            except (json.JSONDecodeError, OSError, ValueError) as error:
                st.error(f"Could not read VLM output: {error}")
        else:
            st.info(
                "VLM analysis file not found. Expected: "
                f"{ADVISORY_VLM_PATH}"
            )

    except FileNotFoundError as error:
        st.error(f"Advisory data file not found: {error}")
        st.info(
            "Check the data/Advisories and data/processed folders "
            "inside your UrbanPulseAI project."
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        st.error(f"Could not load advisory information: {error}")

# =====================================================
# 12. WEATHER INTELLIGENCE TAB
# =====================================================

with weather_tab:
    st.subheader("Hyderabad Weather Intelligence")
    st.caption(
        "Weather measurements from the saved project snapshot. "
        "This page does not fetch live weather conditions."
    )

    try:
        weather_data = load_json_file(str(WEATHER_JSON_PATH))
        current_weather = weather_data.get("current", {})

        if not isinstance(current_weather, dict) or not current_weather:
            st.error(
                "The weather JSON does not contain a populated 'current' object."
            )
        else:
            required_fields = {
                "temperature_2m": ("Temperature", "°C"),
                "relative_humidity_2m": ("Relative humidity", "%"),
                "precipitation": ("Precipitation", "mm"),
                "wind_speed_10m": ("Wind speed", "km/h"),
            }

            missing_fields = [
                field for field in required_fields
                if field not in current_weather
            ]

            if missing_fields:
                st.error(
                    "The weather snapshot is missing required measurements: "
                    + ", ".join(missing_fields)
                )
            else:
                timezone_name = weather_data.get(
                    "timezone", "Asia/Kolkata"
                )
                recorded_time_raw = current_weather.get("time")

                recorded_time = None
                snapshot_age_text = "Snapshot timestamp unavailable."

                if recorded_time_raw:
                    try:
                        recorded_time = datetime.fromisoformat(
                            str(recorded_time_raw)
                        )

                        try:
                            weather_timezone = ZoneInfo(timezone_name)
                        except ZoneInfoNotFoundError:
                            weather_timezone = None

                        if recorded_time.tzinfo is None and weather_timezone:
                            recorded_time = recorded_time.replace(
                                tzinfo=weather_timezone
                            )

                        if recorded_time.tzinfo is not None:
                            now_at_snapshot_timezone = datetime.now(
                                recorded_time.tzinfo
                            )
                            elapsed = (
                                now_at_snapshot_timezone - recorded_time
                            )

                            if elapsed.total_seconds() >= 0:
                                total_hours = int(
                                    elapsed.total_seconds() // 3600
                                )
                                if total_hours < 48:
                                    snapshot_age_text = (
                                        f"Recorded approximately "
                                        f"{total_hours} hours ago."
                                    )
                                else:
                                    age_days = elapsed.days
                                    snapshot_age_text = (
                                        f"Recorded approximately "
                                        f"{age_days} days ago."
                                    )
                            else:
                                snapshot_age_text = (
                                    "The snapshot timestamp is later than "
                                    "the current time; please verify the source."
                                )

                        else:
                            snapshot_age_text = (
                                "Snapshot time is available, but its timezone "
                                "could not be resolved."
                            )

                    except (TypeError, ValueError):
                        snapshot_age_text = (
                            f"Unrecognized snapshot timestamp: "
                            f"{recorded_time_raw}"
                        )

                st.warning(
                    "Saved weather snapshot — NOT live weather. "
                    "Do not use these readings as current outdoor conditions."
                )

                if recorded_time is not None:
                    st.markdown("#### Snapshot details")
                    date_col, timezone_col = st.columns(2)

                    date_col.metric(
                        "Recorded at",
                        recorded_time.strftime("%d %b %Y, %I:%M %p"),
                    )
                    timezone_col.metric(
                        "Timezone",
                        timezone_name,
                    )

                    st.caption(snapshot_age_text)

                st.divider()
                st.markdown("#### Recorded weather measurements")

                temperature = float(current_weather["temperature_2m"])
                humidity = float(current_weather["relative_humidity_2m"])
                precipitation = float(current_weather["precipitation"])
                wind_speed = float(current_weather["wind_speed_10m"])

                weather_col1, weather_col2, weather_col3, weather_col4 = (
                    st.columns(4)
                )

                weather_col1.metric(
                    "🌡️ Temperature",
                    f"{temperature:.1f} °C",
                )
                weather_col2.metric(
                    "💧 Relative humidity",
                    f"{humidity:.0f} %",
                )
                weather_col3.metric(
                    "🌧️ Precipitation",
                    f"{precipitation:.1f} mm",
                )
                weather_col4.metric(
                    "💨 Wind speed",
                    f"{wind_speed:.1f} km/h",
                )

                st.divider()
                st.markdown("#### Reading the snapshot")

                note_col1, note_col2 = st.columns(2)

                with note_col1:
                    if precipitation == 0:
                        st.info(
                            "No precipitation was recorded at the snapshot "
                            "time. This does not establish whether it is "
                            "raining now."
                        )
                    else:
                        st.info(
                            f"The snapshot recorded {precipitation:.1f} mm "
                            "of precipitation for its measurement interval."
                        )

                with note_col2:
                    st.info(
                        "These are individual recorded measurements, not a "
                        "weather trend. A time-series chart requires multiple "
                        "timestamped weather observations."
                    )

                with st.expander("Weather source metadata"):
                    metadata = {
                        key: value
                        for key, value in weather_data.items()
                        if key != "current"
                    }
                    st.json(metadata)

                    st.markdown("**Raw current measurements**")
                    st.json(current_weather)

    except FileNotFoundError:
        st.error(
            "Weather JSON file not found. Expected location: "
            f"{WEATHER_JSON_PATH}"
        )
        st.info(
            "Check that data/Weather/hyderabad_weather.json exists "
            "inside your UrbanPulseAI project."
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        st.error(f"Could not load weather information: {error}")
