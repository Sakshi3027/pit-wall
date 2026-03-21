import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.data.ergast_client import (
    get_driver_standings, get_constructor_standings,
    get_season_schedule, get_historical_results
)
from app.data.fastf1_client import get_lap_times, get_race_results
from app.models.race_predictor import train_model, load_model
from app.models.season_simulator import simulate_season, build_driver_strengths

st.set_page_config(
    page_title="Pit Wall",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size:2rem; font-weight:700; color:#e10600;}
    .sub-header {font-size:1rem; color:#888; margin-top:-10px;}
    .metric-card {background:#1a1a1a; padding:1rem; border-radius:8px; border-left:3px solid #e10600;}
    [data-testid="stSidebar"] {background:#0f0f0f;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ Pit Wall")
    st.markdown("---")
    page = st.radio("Navigate", [
        "Live Standings",
        "Race Analysis",
        "Race Predictor",
        "Season Championship",
    ])
    st.markdown("---")
    season_year = st.selectbox("Season", [2024, 2023, 2022, 2021], index=0)

# ── Page: Live Standings ──────────────────────────────────────────────────────
if page == "Live Standings":
    st.markdown('<p class="main-header">Driver & Constructor Standings</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{season_year} Formula 1 World Championship</p>', unsafe_allow_html=True)

    with st.spinner("Loading standings..."):
        drivers = get_driver_standings(season_year)
        constructors = get_constructor_standings(season_year)
        schedule = get_season_schedule(season_year)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Championship Leader", drivers.iloc[0]["full_name"])
    with col2:
        st.metric("Leader Points", int(drivers.iloc[0]["points"]))
    with col3:
        gap = int(drivers.iloc[0]["points"] - drivers.iloc[1]["points"])
        st.metric("Gap to P2", f"+{gap} pts")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Drivers Championship")
        fig = px.bar(
            drivers.head(10),
            x="points", y="full_name",
            orientation="h",
            color="constructor",
            text="points",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            height=400,
            xaxis_title="Points",
            yaxis_title="",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Constructors Championship")
        fig2 = px.bar(
            constructors,
            x="points", y="constructor",
            orientation="h",
            color="constructor",
            text="points",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig2.update_layout(
            yaxis={"categoryorder": "total ascending"},
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=400,
            xaxis_title="Points",
            yaxis_title="",
        )
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Driver Standings")
    display_df = drivers[["position","full_name","constructor","points","wins"]].copy()
    display_df.columns = ["Pos", "Driver", "Constructor", "Points", "Wins"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Season Schedule")
    st.dataframe(schedule, use_container_width=True, hide_index=True)


# ── Page: Race Analysis ───────────────────────────────────────────────────────
elif page == "Race Analysis":
    st.markdown('<p class="main-header">Race Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Lap times, tire strategy & pace analysis</p>', unsafe_allow_html=True)

    schedule = get_season_schedule(season_year)
    gp_options = schedule["gp_name"].tolist()
    gp_name_map = {
        "Bahrain Grand Prix": "Bahrain",
        "Saudi Arabian Grand Prix": "Saudi Arabia",
        "Australian Grand Prix": "Australia",
        "Japanese Grand Prix": "Japan",
        "Chinese Grand Prix": "China",
        "Miami Grand Prix": "Miami",
        "Emilia Romagna Grand Prix": "Emilia Romagna",
        "Monaco Grand Prix": "Monaco",
        "Canadian Grand Prix": "Canada",
        "Spanish Grand Prix": "Spain",
        "Austrian Grand Prix": "Austria",
        "British Grand Prix": "Great Britain",
        "Hungarian Grand Prix": "Hungary",
        "Belgian Grand Prix": "Belgium",
        "Dutch Grand Prix": "Netherlands",
        "Italian Grand Prix": "Monza",
        "Azerbaijan Grand Prix": "Azerbaijan",
        "Singapore Grand Prix": "Singapore",
        "United States Grand Prix": "United States",
        "Mexico City Grand Prix": "Mexico",
        "São Paulo Grand Prix": "São Paulo",
        "Las Vegas Grand Prix": "Las Vegas",
        "Qatar Grand Prix": "Qatar",
        "Abu Dhabi Grand Prix": "Abu Dhabi",
    }

    col1, col2 = st.columns(2)
    with col1:
        selected_gp = st.selectbox("Grand Prix", gp_options)
    with col2:
        session_type = st.selectbox("Session", ["R", "Q", "FP1", "FP2", "FP3"],
                                     format_func=lambda x: {
                                         "R": "Race", "Q": "Qualifying",
                                         "FP1": "Practice 1", "FP2": "Practice 2",
                                         "FP3": "Practice 3"
                                     }[x])

    gp_key = gp_name_map.get(selected_gp, selected_gp.replace(" Grand Prix", ""))

    if st.button("Load Session Data", type="primary"):
        with st.spinner(f"Loading {selected_gp} {season_year} data..."):
            try:
                laps = get_lap_times(season_year, gp_key, session_type)
                results = get_race_results(season_year, gp_key)
                st.session_state["laps"] = laps
                st.session_state["results"] = results
                st.success(f"Loaded {len(laps)} laps from {selected_gp}")
            except Exception as e:
                st.error(f"Could not load data: {e}")

    if "laps" in st.session_state:
        laps = st.session_state["laps"]
        results = st.session_state["results"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Laps", len(laps))
        with col2:
            fastest = laps.loc[laps["LapTimeSeconds"].idxmin()]
            st.metric("Fastest Lap", f"{fastest['LapTimeSeconds']:.3f}s")
        with col3:
            st.metric("Driver", fastest["Driver"])

        st.markdown("---")
        st.subheader("Lap Time Evolution")
        drivers_available = sorted(laps["Driver"].unique())
        selected_drivers = st.multiselect(
            "Select drivers", drivers_available,
            default=drivers_available[:5]
        )

        if selected_drivers:
            filtered = laps[laps["Driver"].isin(selected_drivers)]
            fig = px.line(
                filtered, x="LapNumber", y="LapTimeSeconds",
                color="Driver", markers=False,
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                xaxis_title="Lap",
                yaxis_title="Lap Time (s)",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tire Strategy")
        compound_colors = {
            "SOFT": "#FF3333", "MEDIUM": "#FFF200",
            "HARD": "#EBEBEB", "INTERMEDIATE": "#39B54A", "WET": "#0067FF"
        }
        fig2 = px.scatter(
            laps, x="LapNumber", y="Driver",
            color="Compound", size_max=6,
            color_discrete_map=compound_colors,
            opacity=0.8,
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=500,
            xaxis_title="Lap",
            yaxis_title="",
        )
        st.plotly_chart(fig2, use_container_width=True)

        if session_type == "R":
            st.subheader("Race Results")
            st.dataframe(
                results[["Abbreviation","TeamName","Position","GridPosition","Points","Status"]],
                use_container_width=True, hide_index=True
            )


# ── Page: Race Predictor ──────────────────────────────────────────────────────
elif page == "Race Predictor":
    st.markdown('<p class="main-header">Race Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML-powered podium probability for the next race</p>', unsafe_allow_html=True)

    st.info("This model is trained on 2021-2023 data and predicts podium probability based on grid position, recent form, circuit history and constructor performance.")

    if st.button("Train / Refresh Model", type="primary"):
        with st.spinner("Fetching training data (2021-2023)..."):
            df = get_historical_results(2021, 2023)
        with st.spinner("Training XGBoost model..."):
            model = train_model(df)
            st.session_state["model"] = model
        st.success("Model trained successfully!")

    try:
        if "model" not in st.session_state:
            model = load_model()
            st.session_state["model"] = model
            st.info("Loaded existing model from disk.")
    except:
        st.warning("No trained model found. Click 'Train / Refresh Model' first.")
        st.stop()

    model = st.session_state["model"]
    standings = get_driver_standings(season_year)

    st.subheader("Configure Next Race")
    col1, col2 = st.columns(2)
    with col1:
        round_num = st.slider("Round number", 1, 24, 12)
    with col2:
        circuit_type = st.selectbox(
            "Circuit type",
            ["high_downforce", "street", "power", "technical"],
            index=0
        )

    circuit_type_map = {"high_downforce": 0, "street": 3, "power": 2, "technical": 4}
    circuit_code = circuit_type_map.get(circuit_type, 0)

    st.subheader("Set Grid Positions")
    grid_data = []
    drivers_list = standings["driver"].tolist()[:10]

    cols = st.columns(5)
    grids = {}
    for i, driver in enumerate(drivers_list):
        with cols[i % 5]:
            grids[driver] = st.number_input(
                driver, min_value=1, max_value=20,
                value=i+1, key=f"grid_{driver}"
            )

    if st.button("Predict Podium", type="primary"):
        from app.models.feature_engineering import build_training_features
        from app.models.race_predictor import FEATURES

        hist = get_historical_results(2021, season_year-1)
        feat_df = build_training_features(hist)

        rows = []
        for driver in drivers_list:
            driver_hist = feat_df[feat_df["driver"] == driver]
            if len(driver_hist) == 0:
                row = {f: 0.0 for f in FEATURES}
            else:
                row = driver_hist.iloc[-1][FEATURES].to_dict()
            row["driver"] = driver
            row["grid"] = grids[driver]
            row["grid_squared"] = grids[driver] ** 2
            row["circuit_type_code"] = circuit_code
            row["round"] = round_num
            row["year"] = season_year
            rows.append(row)

        race_df = pd.DataFrame(rows)
        from app.models.race_predictor import predict_race
        predictions = predict_race(model, race_df)

        st.subheader("Podium Predictions")
        fig = px.bar(
            predictions.head(10),
            x="driver", y="podium_probability",
            color="podium_probability",
            color_continuous_scale=["#333", "#e10600"],
            text=predictions.head(10)["podium_probability"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            xaxis_title="Driver",
            yaxis_title="Podium Probability",
            showlegend=False,
            coloraxis_showscale=False,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            predictions[["driver","podium_probability","predicted_position"]].head(10),
            use_container_width=True, hide_index=True
        )


# ── Page: Season Championship ─────────────────────────────────────────────────
elif page == "Season Championship":
    st.markdown('<p class="main-header">Season Championship Forecast</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Monte Carlo simulation — 10,000 season scenarios</p>', unsafe_allow_html=True)

    with st.spinner("Loading standings..."):
        standings = get_driver_standings(season_year)
        constructors = get_constructor_standings(season_year)
        schedule = get_season_schedule(season_year)

    col1, col2, col3 = st.columns(3)
    total_rounds = len(schedule)
    with col1:
        current_round = st.slider("Rounds completed", 1, total_rounds, min(10, total_rounds))
    with col2:
        remaining = total_rounds - current_round
        st.metric("Races remaining", remaining)
    with col3:
        n_sims = st.selectbox("Simulations", [1000, 5000, 10000], index=1)

    noise = st.slider("Upset factor (higher = more unpredictable season)", 0.05, 0.40, 0.18)

    if st.button("Run Championship Simulation", type="primary"):
        with st.spinner(f"Running {n_sims:,} season simulations..."):
            round_standings = get_driver_standings(season_year, round_num=current_round)
            strengths = build_driver_strengths(round_standings)

            for d in strengths:
                strengths[d] = strengths[d]

            from app.models.season_simulator import simulate_race
            import app.models.season_simulator as sim_module
            sim_module.simulate_race.__globals__

            results = simulate_season(
                current_standings=round_standings,
                remaining_races=remaining,
                driver_strengths=strengths,
                n_simulations=n_sims,
            )
            st.session_state["sim_results"] = results
            st.session_state["sim_standings"] = round_standings

    if "sim_results" in st.session_state:
        results = st.session_state["sim_results"]
        round_standings = st.session_state["sim_standings"]

        st.markdown("---")
        st.subheader("WDC Win Probability")

        top = results[results["wdc_probability"] > 0]
        fig = px.bar(
            top, x="driver", y="wdc_probability",
            color="wdc_probability",
            color_continuous_scale=["#222", "#e10600"],
            text=top["wdc_probability"].apply(lambda x: f"{x}%"),
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            xaxis_title="Driver",
            yaxis_title="Championship Win Probability (%)",
            coloraxis_showscale=False,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Current vs Projected Points")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="Current",
                x=results["driver"],
                y=results["current_points"],
                marker_color="#555",
            ))
            fig2.add_trace(go.Bar(
                name="Projected Final",
                x=results["driver"],
                y=results["avg_final_points"],
                marker_color="#e10600",
            ))
            fig2.update_layout(
                barmode="group",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=350,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_r:
            st.subheader("Points gap to leader")
            leader_pts = round_standings.iloc[0]["points"]
            round_standings["gap"] = leader_pts - round_standings["points"]
            fig3 = px.bar(
                round_standings.head(10),
                x="driver", y="gap",
                color="gap",
                color_continuous_scale=["#e10600", "#333"],
                text="gap",
            )
            fig3.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=350,
                coloraxis_showscale=False,
                yaxis_title="Points behind leader",
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.subheader("Full simulation results")
        st.dataframe(results, use_container_width=True, hide_index=True)