import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Coffee Capsules — Demand Model",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RAW = pd.read_csv("coffee_capsules_data.csv")
HOUSEHOLDS = {
    "Household 1": ("HH1_Regular", "HH1_Premium"),
    "Household 2": ("HH2_Regular", "HH2_Premium"),
    "Household 3": ("HH3_Regular", "HH3_Premium"),
}
CHOICES = ["Regular", "Premium", "No Purchase"]
COLORS = {"Regular": "#1677ff", "Premium": "#ff8a1f", "No Purchase": "#aab4c0"}
BASELINE_R, BASELINE_P = 42, 93


def build_observations():
    rows = []
    for household, (regular_col, premium_col) in HOUSEHOLDS.items():
        for _, row in RAW.iterrows():
            regular_qty = int(row[regular_col])
            premium_qty = int(row[premium_col])
            choice = (
                "Regular" if regular_qty > 0
                else "Premium" if premium_qty > 0
                else "No Purchase"
            )
            rows.append({
                "Household": household,
                "T": int(row["T"]),
                "P_Regular": float(row["P_Regular"]),
                "P_Premium": float(row["P_Premium"]),
                "Regular": regular_qty,
                "Premium": premium_qty,
                "Choice": choice,
                "Quantity": regular_qty + premium_qty,
            })
    return pd.DataFrame(rows)


OBS = build_observations()
PR_MEAN, PR_STD = OBS["P_Regular"].mean(), OBS["P_Regular"].std(ddof=0)
PP_MEAN, PP_STD = OBS["P_Premium"].mean(), OBS["P_Premium"].std(ddof=0)


def features(frame):
    return np.column_stack([
        (frame["P_Regular"].to_numpy() - PR_MEAN) / PR_STD,
        (frame["P_Premium"].to_numpy() - PP_MEAN) / PP_STD,
        (frame["Household"] == "Household 2").astype(float).to_numpy(),
        (frame["Household"] == "Household 3").astype(float).to_numpy(),
    ])


purchase_model = LogisticRegression(C=1, max_iter=5000).fit(
    features(OBS),
    (OBS["Choice"] != "No Purchase").astype(int),
)
BUYERS = OBS[OBS["Choice"] != "No Purchase"]
choice_model = LogisticRegression(C=1, max_iter=5000).fit(
    features(BUYERS),
    (BUYERS["Choice"] == "Premium").astype(int),
)


def probabilities(household, regular_price, premium_price):
    frame = pd.DataFrame({
        "Household": [household],
        "P_Regular": [regular_price],
        "P_Premium": [premium_price],
    })
    buy = purchase_model.predict_proba(features(frame))[0, 1]
    premium_given_buy = choice_model.predict_proba(features(frame))[0, 1]
    return {
        "No Purchase": 1 - buy,
        "Regular": buy * (1 - premium_given_buy),
        "Premium": buy * premium_given_buy,
        "Buy": buy,
        "Premium | Buy": premium_given_buy,
    }


def boundary(model, household):
    coef = model.coef_[0]
    household_effect = (
        coef[2] if household == "Household 2"
        else coef[3] if household == "Household 3"
        else 0
    )
    if abs(coef[1]) < 1e-12:
        return np.nan, np.nan
    slope_z = -coef[0] / coef[1]
    intercept_z = -(model.intercept_[0] + household_effect) / coef[1]
    slope = slope_z * PP_STD / PR_STD
    intercept = PP_MEAN + PP_STD * intercept_z - slope * PR_MEAN
    return float(slope), float(intercept)


def decision_map(household, regular_price, premium_price):
    purchase_slope, purchase_intercept = boundary(purchase_model, household)
    choice_slope, choice_intercept = boundary(choice_model, household)

    xs = np.linspace(30, 70, 180)
    ys = np.linspace(55, 145, 180)
    xx, yy = np.meshgrid(xs, ys)

    grid = pd.DataFrame({
        "Household": household,
        "P_Regular": xx.ravel(),
        "P_Premium": yy.ravel(),
    })
    buy = purchase_model.predict_proba(features(grid))[:, 1].reshape(xx.shape)
    premium = choice_model.predict_proba(features(grid))[:, 1].reshape(xx.shape)
    region = np.where(buy < 0.5, 2, np.where(premium >= 0.5, 1, 0))

    fig = go.Figure(go.Heatmap(
        x=xs, y=ys, z=region, zmin=0, zmax=2,
        colorscale=[
            [0.00, "rgba(22,119,255,.12)"], [0.33, "rgba(22,119,255,.12)"],
            [0.34, "rgba(255,138,31,.10)"], [0.66, "rgba(255,138,31,.10)"],
            [0.67, "rgba(170,180,192,.12)"], [1.00, "rgba(170,180,192,.12)"],
        ], showscale=False, hoverinfo="skip",
    ))

    for slope, intercept, name, dash, color in [
        (purchase_slope, purchase_intercept, "Purchase boundary", "dash", "#1677ff"),
        (choice_slope, choice_intercept, "Regular / Premium", "dash", "#ff8a1f"),
    ]:
        if np.isfinite(slope) and np.isfinite(intercept):
            line = slope * xs + intercept
            mask = np.isfinite(line) & (line >= ys.min()) & (line <= ys.max())
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=xs[mask], y=line[mask], mode="lines", name=name,
                    line={"color": color, "width": 2, "dash": dash},
                ))

    household_data = OBS[OBS["Household"] == household]
    for choice in CHOICES:
        data = household_data[household_data["Choice"] == choice]
        if data.empty:
            continue
        fig.add_trace(go.Scatter(
            x=data["P_Regular"], y=data["P_Premium"], mode="markers", name=choice,
            marker={"size": 10, "color": COLORS[choice], "line": {"color": "white", "width": 1.5}},
            customdata=data[["T", "Quantity"]].to_numpy(),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>Week %{customdata[0]}<br>"
                "Regular €%{x:.0f}<br>Premium €%{y:.0f}<br>"
                "Quantity %{customdata[1]}<extra></extra>"
            ),
        ))

    scenario = probabilities(household, regular_price, premium_price)
    predicted = max(CHOICES, key=scenario.get)
    fig.add_trace(go.Scatter(
        x=[regular_price], y=[premium_price], mode="markers", name="Scenario",
        marker={"symbol": "star", "size": 18, "color": "#172033", "line": {"color": "white", "width": 2}},
    ))

    fig.update_layout(
        height=500, margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Inter, Arial, sans-serif", "color": "#172033"},
        xaxis={"title": "Regular price (€)", "range": [30, 70], "gridcolor": "#e8edf3", "zeroline": False, "fixedrange": True},
        yaxis={"title": "Premium price (€)", "range": [55, 145], "gridcolor": "#e8edf3", "zeroline": False, "fixedrange": True},
        legend={"orientation": "h", "y": 1.04, "x": 0}, hoverlabel={"bgcolor": "white"},
    )
    return fig, scenario, predicted, (purchase_slope, purchase_intercept), (choice_slope, choice_intercept)


def quantity_stats(frame):
    regular = frame.loc[frame["Choice"] == "Regular", "Regular"]
    premium = frame.loc[frame["Choice"] == "Premium", "Premium"]
    return float(frame["Quantity"].mean()), float(regular.mean()) if len(regular) else 0.0, float(premium.mean()) if len(premium) else 0.0


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --ink:#101828; --muted:#667085; --line:#e5eaf0; --page:#f3f5f7; --blue:#1677ff; --orange:#ff8a1f; --green:#12b76a; }
html,body,[class*="css"],.stApp { font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important; }
.stApp,[data-testid="stAppViewContainer"] { background:var(--page); color:var(--ink); }
[data-testid="stSidebar"] { display:none; }
.block-container { max-width:1460px; padding:24px 28px 54px; }
.top-shell { background:rgba(255,255,255,.96); border:1px solid var(--line); border-radius:24px; padding:14px 18px; margin-bottom:26px; box-shadow:0 10px 30px rgba(16,24,40,.035); }
.brand { display:flex; align-items:center; gap:10px; font-size:18px; font-weight:700; color:#172033; }
.brand-icon { width:28px; height:28px; border-radius:8px; display:grid; place-items:center; color:white; background:linear-gradient(145deg,#ffb21a,#ff8a1f); font-size:16px; }
.nav { display:flex; gap:26px; align-items:center; justify-content:center; }
.nav a { color:#344054; text-decoration:none; font-size:13px; }
.nav a.active { background:#172033; color:white; padding:9px 16px; border-radius:11px; box-shadow:0 4px 12px rgba(16,24,40,.12); }
.profile { width:34px; height:34px; border:1px solid var(--line); border-radius:50%; display:grid; place-items:center; color:#344054; }
.eyebrow { display:inline-block; padding:7px 11px; border-radius:999px; background:#edf3f9; color:#4b6380; font-size:11px; font-weight:600; margin-bottom:10px; }
.hero { display:flex; justify-content:space-between; gap:24px; align-items:flex-end; margin:4px 0 24px; }
.hero-copy { max-width:760px; }
.hero h1 { font-size:38px; line-height:1.05; letter-spacing:-1.8px; margin:0 0 12px; color:#101828; font-weight:600; }
.hero p { margin:0; color:#667085; font-size:14px; line-height:1.6; }
.controls { min-width:320px; background:#fff; border:1px solid var(--line); border-radius:16px; padding:10px 12px; }
.section-label { color:#667085; font-size:11px; font-weight:600; margin:0 0 7px; }
.card { background:#fff; border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 5px 20px rgba(16,24,40,.025); height:100%; }
.card-title { color:#101828; font-size:13px; font-weight:600; margin-bottom:5px; }
.card-sub { color:#98a2b3; font-size:11px; line-height:1.5; }
.kpi-value { color:#101828; font-size:26px; font-weight:600; letter-spacing:-1px; margin-top:6px; }
.insight { background:linear-gradient(145deg,#172033,#223a53); color:#fff; border-radius:18px; padding:20px; min-height:180px; box-shadow:0 14px 30px rgba(23,32,51,.14); }
.insight .eyebrow { background:rgba(255,255,255,.10); color:#dbeafe; }
.insight h2 { color:white; font-size:27px; margin:5px 0; letter-spacing:-1px; }
.insight p { color:#cbd5e1; font-size:12px; margin:0; }
.prob-row { display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid rgba(255,255,255,.10); font-size:12px; }
.prob-row:last-child { border-bottom:0; }
.pill { display:inline-block; padding:5px 9px; border-radius:999px; font-size:10px; font-weight:600; background:#f2f4f7; color:#475467; margin:2px 2px 0 0; }
.equation { background:#f8fafc; border:1px solid #e7edf3; border-radius:12px; padding:11px 13px; margin-top:9px; color:#172033; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; }
.info { background:#f5f9ff; border:1px solid #dcecff; border-radius:14px; padding:13px 15px; color:#475467; font-size:11px; line-height:1.5; }
.warning { background:#fff8ed; border:1px solid #f6d9ad; border-radius:14px; padding:13px 15px; color:#7a4a0b; font-size:11px; line-height:1.5; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:14px 15px; box-shadow:0 4px 16px rgba(16,24,40,.025); }
[data-testid="stMetricLabel"] { font-size:11px!important; color:#667085!important; }
[data-testid="stMetricValue"] { font-size:25px!important; color:#101828!important; }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:14px; background:#fff; }
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; }
.stSelectbox label,.stSlider label { color:#667085!important; font-size:11px!important; }
@media(max-width:900px) { .block-container{padding:16px 16px 38px}.hero{align-items:stretch;flex-direction:column}.controls{min-width:0}.nav{gap:13px}.hero h1{font-size:32px;} }
@media(max-width:640px) { .block-container{padding:10px 10px 30px}.top-shell{border-radius:18px;padding:12px}.nav{display:none}.brand{font-size:16px}.hero h1{font-size:27px;letter-spacing:-1px}.hero p{font-size:13px}.card{border-radius:15px;padding:14px}.insight{border-radius:15px}[data-testid="stMetric"]{padding:11px 12px}[data-testid="stMetricValue"]{font-size:21px!important}.js-plotly-plot .plotly{width:100%!important;} }
</style>
""", unsafe_allow_html=True)

household_col, regular_col, premium_col = st.columns([1.1, 0.8, 0.8])
with household_col:
    household = st.selectbox("Household", list(HOUSEHOLDS), label_visibility="collapsed")
with regular_col:
    regular_price = st.slider("Regular price", 30, 70, BASELINE_R, key="regular_price")
with premium_col:
    premium_price = st.slider("Premium price", 65, 105, BASELINE_P, key="premium_price")

fig, scenario, predicted, purchase_boundary, choice_boundary = decision_map(household, regular_price, premium_price)
avg_qty, regular_qty, premium_qty = quantity_stats(OBS[OBS["Household"] == household])

st.markdown("""
<div class="top-shell"><div style="display:flex;align-items:center;justify-content:space-between;gap:18px;">
<div class="brand"><span class="brand-icon">☕</span> Coffee Capsule Lab</div>
<div class="nav"><a class="active" href="#home">Home</a><a href="#model">Model</a><a href="#data">Data</a><a href="#methodology">Methodology</a><a href="#insights">Insights</a></div>
<div class="profile">◦</div></div></div><div id="home"></div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero"><div class="hero-copy"><div class="eyebrow">Two-stage logistic classifier</div>
<h1>Coffee Capsules<br>Purchase & Choice Analysis</h1>
<p>Explore how Regular and Premium prices change the estimated probability of purchasing, choosing Premium, or not purchasing for each household.</p></div>
<div class="controls"><div class="section-label">Current scenario</div><div style="color:#172033;font-size:13px;font-weight:600;">Regular €{regular_price} &nbsp;·&nbsp; Premium €{premium_price}</div>
<div class="card-sub" style="margin-top:5px;">Adjust the controls above to test another price pair.</div></div></div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Purchase probability", f"{scenario['Buy']:.0%}")
m2.metric("P(Regular)", f"{scenario['Regular']:.0%}")
m3.metric("P(Premium)", f"{scenario['Premium']:.0%}")
m4.metric("P(No Purchase)", f"{scenario['No Purchase']:.0%}")

st.markdown('<div id="model"></div>', unsafe_allow_html=True)
left, right = st.columns([1.65, 0.8], gap="medium")
with left:
    st.markdown('<div class="card"><div class="card-title">Decision boundaries</div><div class="card-sub">Observed weekly choices and the 50% prediction boundaries implied by the two logistic stages.</div></div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
with right:
    st.markdown(f'<div class="insight"><div class="eyebrow">Scenario result</div><h2>{predicted}</h2><p>At Regular €{regular_price} and Premium €{premium_price}</p><div style="height:12px"></div><div class="prob-row"><span>Regular</span><b>{scenario["Regular"]:.1%}</b></div><div class="prob-row"><span>Premium</span><b>{scenario["Premium"]:.1%}</b></div><div class="prob-row"><span>No Purchase</span><b>{scenario["No Purchase"]:.1%}</b></div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="card"><div class="card-title">Demand snapshot</div><div class="card-sub">Observed quantity across the 11 weeks</div><div class="kpi-value">{avg_qty:.2f}</div><div class="card-sub">capsules / week</div><div style="height:8px"></div><span class="pill">Regular when purchased&nbsp; {regular_qty:.2f}</span><span class="pill">Premium when purchased&nbsp; {premium_qty:.2f}</span></div>', unsafe_allow_html=True)

st.markdown('<div id="insights"></div>', unsafe_allow_html=True)
i1, i2, i3 = st.columns(3)
with i1:
    st.markdown('<div class="card"><div class="card-title">Purchase stage</div><div class="card-sub">Estimates P(Buy) using Regular price, Premium price, and household indicators.</div><div class="equation">logit(P(Buy)) = β₀ + βᵣzᵣ + βₚzₚ + household effects</div></div>', unsafe_allow_html=True)
with i2:
    st.markdown('<div class="card"><div class="card-title">Choice stage</div><div class="card-sub">Among observed buyers, estimates P(Premium | Buy).</div><div class="equation">logit(P(Premium | Buy)) = γ₀ + γᵣzᵣ + γₚzₚ + household effects</div></div>', unsafe_allow_html=True)
with i3:
    st.markdown(f'<div class="card"><div class="card-title">Current household</div><div class="card-sub">{household} · 11 weekly observations</div><div class="kpi-value">{max(scenario["Regular"], scenario["Premium"], scenario["No Purchase"]):.0%}</div><div class="card-sub">highest predicted choice probability</div></div>', unsafe_allow_html=True)

st.markdown('<div id="methodology"></div>', unsafe_allow_html=True)
st.subheader("Model & methodology")
with st.expander("How the two-stage model works"):
    st.markdown(f"""**Stage 1 — Purchase.** A pooled logistic regression estimates the probability that the household purchases anything.

**Stage 2 — Product choice.** For observations where a purchase is observed, a second logistic regression estimates the probability of Premium rather than Regular.

The final probabilities are:
- **P(No Purchase) = 1 − P(Buy)**
- **P(Regular) = P(Buy) × [1 − P(Premium | Buy)]**
- **P(Premium) = P(Buy) × P(Premium | Buy)**

For the current scenario, the model predicts **{scenario['Regular']:.1%} Regular, {scenario['Premium']:.1%} Premium, and {scenario['No Purchase']:.1%} No Purchase**.
""")
with st.expander("How should I read the boundaries?"):
    ps, pi = purchase_boundary
    cs, ci = choice_boundary
    st.markdown(f"""The dashed lines are **50% prediction contours**. They are not literal willingness-to-pay curves.

**Purchase boundary**
`P_P = {ps:.2f} × P_R + {pi:.1f}`

**Regular / Premium boundary**
`P_P = {cs:.2f} × P_R + {ci:.1f}`

The background region shows which choice the model predicts as most likely at each price combination.
""")
with st.expander("Important interpretation note"):
    st.markdown("This is an exploratory classification model rather than a structural demand or causal price-elasticity model. The boundary lines are useful for visualizing model predictions, but should not be presented as validated willingness-to-pay thresholds.")

st.markdown('<div id="data"></div>', unsafe_allow_html=True)
st.subheader("Observed weeks")
sample = OBS[OBS["Household"] == household][["T", "P_Regular", "P_Premium", "Choice", "Quantity"]].copy()
sample.columns = ["Week", "Regular price", "Premium price", "Observed choice", "Quantity"]
st.dataframe(sample, use_container_width=True, hide_index=True)
st.caption("33 household-week observations · 3 households · exploratory research model")
