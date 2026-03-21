import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import google.generativeai as genai
from app.data.ergast_client import (
    get_driver_standings, get_constructor_standings,
    get_season_schedule, get_historical_results
)
from app.data.fastf1_client import get_lap_times, get_race_results
from app.models.race_predictor import train_model, load_model
from app.models.season_simulator import simulate_season, build_driver_strengths

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(
    page_title="Pit Wall",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size:2rem; font-weight:700; color:#e10600;}
    .sub-header  {font-size:1rem; color:#888; margin-top:-10px;}
    [data-testid="stSidebar"] {background:#0f0f0f;}
</style>
""", unsafe_allow_html=True)

GP_MAP = {
    "Australian Grand Prix": "Australia",
    "Chinese Grand Prix": "China",
    "Japanese Grand Prix": "Japan",
    "Miami Grand Prix": "Miami",
    "Canadian Grand Prix": "Canada",
    "Monaco Grand Prix": "Monaco",
    "Barcelona Grand Prix": "Spain",
    "Austrian Grand Prix": "Austria",
    "British Grand Prix": "Great Britain",
    "Belgian Grand Prix": "Belgium",
    "Hungarian Grand Prix": "Hungary",
    "Dutch Grand Prix": "Netherlands",
    "Italian Grand Prix": "Monza",
    "Spanish Grand Prix": "Spain",
    "Azerbaijan Grand Prix": "Azerbaijan",
    "Singapore Grand Prix": "Singapore",
    "United States Grand Prix": "United States",
    "Mexico City Grand Prix": "Mexico",
    "São Paulo Grand Prix": "São Paulo",
    "Brazilian Grand Prix": "São Paulo",
    "Las Vegas Grand Prix": "Las Vegas",
    "Qatar Grand Prix": "Qatar",
    "Abu Dhabi Grand Prix": "Abu Dhabi",
    "Bahrain Grand Prix": "Bahrain",
    "Saudi Arabian Grand Prix": "Saudi Arabia",
    "Emilia Romagna Grand Prix": "Emilia Romagna",
    "Emilia-Romagna Grand Prix": "Emilia Romagna",
}

def races_completed(schedule_df):
    today = str(pd.Timestamp.now().date())
    return len([r for r in schedule_df["date"] if r <= today])

def build_f1_context(season_year=2026) -> str:
    try:
        standings    = get_driver_standings(season_year)
        constructors = get_constructor_standings(season_year)
        schedule     = get_season_schedule(season_year)
        done         = races_completed(schedule)

        driver_str = "\n".join([
            f"  P{row['position']} {row['full_name']} ({row['constructor']}): "
            f"{int(row['points'])} pts, {int(row['wins'])} wins"
            for _, row in standings.iterrows()
        ])
        constructor_str = "\n".join([
            f"  P{row['position']} {row['constructor']}: {int(row['points'])} pts"
            for _, row in constructors.iterrows()
        ])
        schedule_str = "\n".join([
            f"  R{row['round']} {row['gp_name']} — {row['date']} ({row['country']})"
            for _, row in schedule.iterrows()
        ])

        return f"""You are an expert F1 Race Engineer and data analyst for the Pit Wall app.
You have access to LIVE {season_year} F1 season data — {done} races completed so far.

=== {season_year} DRIVER STANDINGS (LIVE) ===
{driver_str}

=== {season_year} CONSTRUCTOR STANDINGS (LIVE) ===
{constructor_str}

=== {season_year} SEASON SCHEDULE ===
{schedule_str}

=== YOUR ROLE ===
- Analyze championship standings and title fights using REAL data above
- Discuss race strategy: tire compounds, pit windows, undercuts, overcuts
- Compare driver and constructor performance with actual numbers
- Predict race outcomes based on current form
- Explain F1 regulations and technical concepts clearly
- Give confident, data-driven analysis like a real race engineer

Always reference actual points and positions from the data above.
Be direct, expert, and insightful. No waffle — real analysis only.
"""
    except Exception as e:
        return f"You are an expert F1 Race Engineer. Use your F1 knowledge to answer questions. (Live data error: {e})"

def get_gemini_response(context: str, messages: list) -> str:
    if not GEMINI_API_KEY:
        return "⚠️ GEMINI_API_KEY not set. Add it to your .env file."
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=context,
    )
    history = []
    for msg in messages[:-1]:
        history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]]
        })
    chat  = model.start_chat(history=history)
    reply = chat.send_message(messages[-1]["content"])
    return reply.text

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏎️ Pit Wall")
    st.markdown("---")
    page = st.radio("Navigate", [
        "Live Standings",
        "Race Analysis",
        "Race Predictor",
        "Season Championship",
        "AI Race Engineer",
    ])
    st.markdown("---")
    season_year = st.selectbox("Season", [2026, 2025, 2024, 2023, 2022], index=0)

    if page == "AI Race Engineer":
        st.markdown("---")
        if st.button("Clear conversation"):
            st.session_state["messages"] = []
            st.rerun()
        st.markdown("**Try asking:**")
        suggestions = [
            "Who will win the 2026 championship?",
            "Why is Verstappen struggling?",
            "Compare Russell vs Antonelli",
            "How does undercut strategy work?",
            "Hamilton at Ferrari — analysis",
            "Which team improved most?",
            "Best race pace right now?",
            "Explain DRS strategy",
        ]
        for s in suggestions:
            if st.button(s, key=f"sug_{s[:20]}"):
                if "messages" not in st.session_state:
                    st.session_state["messages"] = []
                if "f1_context" not in st.session_state:
                    st.session_state["f1_context"] = build_f1_context(season_year)
                st.session_state["messages"].append({"role": "user", "content": s})
                try:
                    reply = get_gemini_response(
                        st.session_state["f1_context"],
                        st.session_state["messages"],
                    )
                except Exception as e:
                    reply = f"⚠️ {'Quota hit — try tomorrow.' if 'quota' in str(e).lower() or '429' in str(e) else str(e)}"
                st.session_state["messages"].append({"role": "assistant", "content": reply})
                st.rerun()

# ── Page: Live Standings ──────────────────────────────────────────────────────
if page == "Live Standings":
    st.markdown('<p class="main-header">Driver & Constructor Standings</p>', unsafe_allow_html=True)

    with st.spinner("Loading standings..."):
        drivers      = get_driver_standings(season_year)
        constructors = get_constructor_standings(season_year)
        schedule     = get_season_schedule(season_year)

    done  = races_completed(schedule)
    total = len(schedule)
    st.markdown(
        f'<p class="sub-header">{season_year} Formula 1 World Championship · '
        f'Round {done} of {total}</p>', unsafe_allow_html=True
    )

    if season_year == 2026:
        gap = int(drivers.iloc[0]["points"] - drivers.iloc[1]["points"])
        st.success(
            f"🔴 LIVE 2026 · {drivers.iloc[0]['full_name']} leads "
            f"{drivers.iloc[1]['full_name']} by {gap} pts after {done} races"
        )
    elif season_year == 2025:
        st.info(f"2025 Final — {drivers.iloc[0]['full_name']} champion · {int(drivers.iloc[0]['points'])} pts")

    c1, c2, c3 = st.columns(3)
    c1.metric("Championship Leader", drivers.iloc[0]["full_name"])
    c2.metric("Leader Points",       int(drivers.iloc[0]["points"]))
    c3.metric("Gap to P2",           f"+{int(drivers.iloc[0]['points'] - drivers.iloc[1]['points'])} pts")

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Drivers Championship")
        fig = px.bar(
            drivers.head(10), x="points", y="full_name",
            orientation="h", color="constructor", text="points",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            yaxis={"categoryorder":"total ascending"},
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=400, xaxis_title="Points", yaxis_title="",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Constructors Championship")
        fig2 = px.bar(
            constructors, x="points", y="constructor",
            orientation="h", color="constructor", text="points",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig2.update_layout(
            yaxis={"categoryorder":"total ascending"},
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=400, xaxis_title="Points", yaxis_title="",
        )
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Driver Standings")
    disp = drivers[["position","full_name","constructor","points","wins"]].copy()
    disp.columns = ["Pos","Driver","Constructor","Points","Wins"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.subheader("Season Schedule")
    st.dataframe(schedule, use_container_width=True, hide_index=True)


# ── Page: Race Analysis ───────────────────────────────────────────────────────
elif page == "Race Analysis":
    st.markdown('<p class="main-header">Race Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Lap times, tire strategy & pace analysis</p>', unsafe_allow_html=True)

    schedule   = get_season_schedule(season_year)
    gp_options = schedule["gp_name"].tolist()

    c1, c2 = st.columns(2)
    with c1:
        selected_gp = st.selectbox("Grand Prix", gp_options)
    with c2:
        session_type = st.selectbox(
            "Session", ["R","Q","FP1","FP2","FP3"],
            format_func=lambda x: {"R":"Race","Q":"Qualifying",
                                   "FP1":"Practice 1","FP2":"Practice 2","FP3":"Practice 3"}[x]
        )

    gp_key = GP_MAP.get(selected_gp, selected_gp.replace(" Grand Prix",""))

    if st.button("Load Session Data", type="primary"):
        with st.spinner(f"Loading {selected_gp} {season_year}..."):
            try:
                laps    = get_lap_times(season_year, gp_key, session_type)
                results = get_race_results(season_year, gp_key)
                st.session_state["laps"]    = laps
                st.session_state["results"] = results
                st.success(f"Loaded {len(laps)} laps from {selected_gp}")
            except Exception as e:
                st.error(f"Could not load data: {e}")

    if "laps" in st.session_state:
        laps    = st.session_state["laps"]
        results = st.session_state["results"]

        fastest = laps.loc[laps["LapTimeSeconds"].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Laps",  len(laps))
        c2.metric("Fastest Lap", f"{fastest['LapTimeSeconds']:.3f}s")
        c3.metric("Driver",      fastest["Driver"])

        st.markdown("---")
        st.subheader("Lap Time Evolution")
        avail = sorted(laps["Driver"].unique())
        sel   = st.multiselect("Select drivers", avail, default=avail[:5])
        if sel:
            fig = px.line(
                laps[laps["Driver"].isin(sel)],
                x="LapNumber", y="LapTimeSeconds", color="Driver",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=400, xaxis_title="Lap", yaxis_title="Lap Time (s)",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tire Strategy")
        fig2 = px.scatter(
            laps, x="LapNumber", y="Driver", color="Compound",
            color_discrete_map={"SOFT":"#FF3333","MEDIUM":"#FFF200",
                                 "HARD":"#EBEBEB","INTERMEDIATE":"#39B54A","WET":"#0067FF"},
            opacity=0.8,
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=500, xaxis_title="Lap", yaxis_title="",
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

    st.info("Trained on 2022-2024 data · predicts podium probability from grid position, recent form, circuit history & constructor pace")

    if st.button("Train / Refresh Model", type="primary"):
        with st.spinner("Fetching training data (2022-2024)..."):
            df = get_historical_results(2022, 2024)
        with st.spinner("Training XGBoost model..."):
            model = train_model(df)
            st.session_state["model"] = model
        st.success("Model trained!")

    try:
        if "model" not in st.session_state:
            model = load_model()
            st.session_state["model"] = model
            st.info("Loaded existing model from disk.")
    except:
        st.warning("No model found — click 'Train / Refresh Model' first.")
        st.stop()

    model     = st.session_state["model"]
    standings = get_driver_standings(season_year)

    st.subheader("Configure Next Race")
    c1, c2 = st.columns(2)
    with c1:
        round_num = st.slider("Round number", 1, 24, 4)
    with c2:
        circuit_type = st.selectbox(
            "Circuit type", ["high_downforce","street","power","technical"]
        )

    circuit_code = {"high_downforce":0,"street":3,"power":2,"technical":4}.get(circuit_type,0)

    st.subheader("Set Grid Positions")
    drivers_list = standings["driver"].tolist()[:10]
    cols  = st.columns(5)
    grids = {}
    for i, driver in enumerate(drivers_list):
        with cols[i % 5]:
            grids[driver] = st.number_input(
                driver, min_value=1, max_value=20, value=i+1, key=f"grid_{driver}"
            )

    if st.button("Predict Podium", type="primary"):
        from app.models.feature_engineering import build_training_features
        from app.models.race_predictor import FEATURES, predict_race

        with st.spinner("Building features..."):
            hist    = get_historical_results(2022, 2024)
            feat_df = build_training_features(hist)

        rows = []
        for driver in drivers_list:
            dh  = feat_df[feat_df["driver"] == driver]
            row = dh.iloc[-1][FEATURES].to_dict() if len(dh) > 0 else {f: 0.0 for f in FEATURES}
            row.update({
                "driver": driver,
                "grid": grids[driver],
                "grid_squared": grids[driver] ** 2,
                "circuit_type_code": circuit_code,
                "round": round_num,
                "year": season_year,
            })
            rows.append(row)

        predictions = predict_race(model, pd.DataFrame(rows))

        st.subheader("Podium Predictions")
        fig = px.bar(
            predictions.head(10), x="driver", y="podium_probability",
            color="podium_probability",
            color_continuous_scale=["#333","#e10600"],
            text=predictions.head(10)["podium_probability"].apply(lambda x: f"{x:.1%}"),
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=400, xaxis_title="Driver", yaxis_title="Podium Probability",
            showlegend=False, coloraxis_showscale=False,
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
    st.markdown('<p class="sub-header">Monte Carlo simulation — who wins the championship?</p>', unsafe_allow_html=True)

    with st.spinner("Loading standings..."):
        schedule     = get_season_schedule(season_year)
        constructors = get_constructor_standings(season_year)

    total_rounds = len(schedule)
    done         = races_completed(schedule)

    if season_year == 2026:
        st.success(f"🔴 Simulating LIVE 2026 season — {done} races done, {total_rounds - done} remaining")

    c1, c2, c3 = st.columns(3)
    with c1:
        current_round = st.slider(
            "Rounds completed", 1, total_rounds,
            min(done, total_rounds) if done > 0 else 1
        )
    with c2:
        remaining = total_rounds - current_round
        st.metric("Races remaining", remaining)
    with c3:
        n_sims = st.selectbox("Simulations", [1000, 5000, 10000], index=1)

    noise = st.slider("Upset factor", 0.05, 0.40, 0.20)

    if st.button("Run Championship Simulation", type="primary"):
        with st.spinner(f"Fetching round {current_round} standings..."):
            round_standings = get_driver_standings(season_year, round_num=current_round)

        if round_standings.empty:
            st.error("No standings data found. Try a lower round number.")
            st.stop()

        strengths = build_driver_strengths(round_standings)

        with st.spinner(f"Running {n_sims:,} season simulations..."):
            import numpy as np
            POINTS = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
            drivers_  = round_standings["driver"].tolist()
            base_pts  = dict(zip(round_standings["driver"], round_standings["points"]))
            win_counts    = {d: 0 for d in drivers_}
            final_pts_sum = {d: 0.0 for d in drivers_}

            for _ in range(n_sims):
                season_pts = base_pts.copy()
                for __ in range(remaining):
                    scores = {
                        d: max(0, s + np.random.normal(0, noise))
                        for d, s in strengths.items()
                    }
                    order = sorted(scores, key=scores.get, reverse=True)
                    for pos, d in enumerate(order, 1):
                        if d in season_pts:
                            season_pts[d] = season_pts.get(d,0) + POINTS.get(pos,0)
                            if pos == 1:
                                season_pts[d] += 1
                champ = max(season_pts, key=season_pts.get)
                win_counts[champ] += 1
                for d in drivers_:
                    final_pts_sum[d] += season_pts.get(d,0)

            results = pd.DataFrame([{
                "driver": d,
                "wdc_probability": round(win_counts[d]/n_sims*100, 1),
                "avg_final_points": round(final_pts_sum[d]/n_sims, 1),
                "current_points": base_pts.get(d, 0),
            } for d in drivers_]).sort_values("wdc_probability", ascending=False).reset_index(drop=True)

        st.session_state["sim_results"]   = results
        st.session_state["sim_standings"] = round_standings

    if "sim_results" in st.session_state:
        results         = st.session_state["sim_results"]
        round_standings = st.session_state["sim_standings"]

        st.markdown("---")
        winner = results.iloc[0]
        st.markdown(
            f"### Model predicts: **{winner['driver']}** wins the "
            f"{season_year} championship ({winner['wdc_probability']}% probability)"
        )

        top = results[results["wdc_probability"] > 0]
        fig = px.bar(
            top, x="driver", y="wdc_probability",
            color="wdc_probability", color_continuous_scale=["#222","#e10600"],
            text=top["wdc_probability"].apply(lambda x: f"{x}%"),
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=420, xaxis_title="Driver",
            yaxis_title="Championship Win Probability (%)",
            coloraxis_showscale=False,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Current vs Projected Final Points")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Current", x=results["driver"],
                                   y=results["current_points"], marker_color="#555"))
            fig2.add_trace(go.Bar(name="Projected Final", x=results["driver"],
                                   y=results["avg_final_points"], marker_color="#e10600"))
            fig2.update_layout(
                barmode="group", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", height=350,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_r:
            st.subheader("Points gap to leader")
            leader_pts    = round_standings.iloc[0]["points"]
            gap_df        = round_standings.copy()
            gap_df["gap"] = leader_pts - gap_df["points"]
            fig3 = px.bar(
                gap_df.head(10), x="driver", y="gap",
                color="gap", color_continuous_scale=["#e10600","#333"], text="gap",
            )
            fig3.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=350, coloraxis_showscale=False,
                yaxis_title="Points behind leader",
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")
        st.subheader("Full simulation breakdown")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.subheader("Constructors Championship")
        st.dataframe(constructors, use_container_width=True, hide_index=True)


# ── Page: AI Race Engineer ────────────────────────────────────────────────────
elif page == "AI Race Engineer":
    st.markdown('<p class="main-header">AI Race Engineer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Gemini 2.0 Flash · Live 2026 F1 data</p>', unsafe_allow_html=True)

    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not set. Add it to .env then restart.")
        st.stop()

    if "f1_context" not in st.session_state:
        with st.spinner("Loading live F1 data..."):
            st.session_state["f1_context"] = build_f1_context(season_year)
        st.success("Live 2026 season data loaded!")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if not st.session_state["messages"]:
        standings = get_driver_standings(2026)
        gap = int(standings.iloc[0]["points"] - standings.iloc[1]["points"])
        st.markdown(f"""
        <div style="background:#1a1a1a;border-left:3px solid #e10600;
                    padding:1rem 1.5rem;border-radius:0 8px 8px 0;margin-bottom:1rem;">
        <b>Race engineer online.</b> Live 2026 data loaded —
        {standings.iloc[0]['full_name']} leads {standings.iloc[1]['full_name']} by {gap} pts.
        Ask me anything about strategy, predictions, or F1.
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask your race engineer..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                try:
                    reply = get_gemini_response(
                        st.session_state["f1_context"],
                        st.session_state["messages"],
                    )
                except Exception as e:
                    reply = f"⚠️ {'Quota hit — try tomorrow.' if 'quota' in str(e).lower() or '429' in str(e) else str(e)}"
            st.markdown(reply)
            st.session_state["messages"].append({"role": "assistant", "content": reply})