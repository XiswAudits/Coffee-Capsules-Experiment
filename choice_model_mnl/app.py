import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="Coffee Capsules — MNL Choice Model", page_icon="☕", layout="wide", initial_sidebar_state="collapsed")

RAW = pd.read_csv("coffee_capsules_data.csv")
HOUSEHOLDS = {"Household 1": ("HH1_Regular", "HH1_Premium"), "Household 2": ("HH2_Regular", "HH2_Premium"), "Household 3": ("HH3_Regular", "HH3_Premium")}
CHOICES = ["Regular", "Premium", "No Purchase"]
COLORS = {"Regular": "#1677ff", "Premium": "#ff8a1f", "No Purchase": "#aab4c0"}
BASELINE_R, BASELINE_P = 42, 93
PARAMETER_BOUNDS = [(-20.0, 20.0)] * 4
PARAMETER_NAMES = ["ASC_Regular", "ASC_Premium", "Beta_Regular_Price", "Beta_Premium_Price"]


def build_observations():
    rows = []
    for household, (regular_col, premium_col) in HOUSEHOLDS.items():
        for _, row in RAW.iterrows():
            regular_qty = int(row[regular_col]); premium_qty = int(row[premium_col])
            choice = "Regular" if regular_qty > 0 else "Premium" if premium_qty > 0 else "No Purchase"
            rows.append({"Household": household, "T": int(row["T"]), "P_Regular": float(row["P_Regular"]), "P_Premium": float(row["P_Premium"]), "Regular": regular_qty, "Premium": premium_qty, "Choice": choice, "Quantity": regular_qty + premium_qty})
    return pd.DataFrame(rows)


OBS = build_observations()
PR_MEAN, PR_STD = OBS["P_Regular"].mean(), OBS["P_Regular"].std(ddof=0)
PP_MEAN, PP_STD = OBS["P_Premium"].mean(), OBS["P_Premium"].std(ddof=0)


def softmax(utilities):
    shifted = utilities - np.max(utilities, axis=1, keepdims=True)
    exp_u = np.exp(shifted)
    return exp_u / exp_u.sum(axis=1, keepdims=True)


def utility(parameters, regular_price, premium_price):
    asc_regular, asc_premium, beta_regular, beta_premium = parameters
    z_regular = (np.asarray(regular_price) - PR_MEAN) / PR_STD
    z_premium = (np.asarray(premium_price) - PP_MEAN) / PP_STD
    return np.column_stack([asc_regular + beta_regular * z_regular, asc_premium + beta_premium * z_premium, np.zeros_like(z_regular)])


def fit_household(household):
    sample = OBS[OBS["Household"] == household].reset_index(drop=True)
    observed = sample["Choice"].map({"Regular": 0, "Premium": 1, "No Purchase": 2}).to_numpy()

    def negative_log_likelihood(parameters):
        probs = softmax(utility(parameters, sample["P_Regular"], sample["P_Premium"]))
        chosen = probs[np.arange(len(observed)), observed]
        return float(-np.log(np.clip(chosen, 1e-15, 1)).sum())

    starts = [np.array([0.0, 0.0, -1.0, -1.0]), np.array([1.0, 0.0, -1.0, -1.0]), np.array([0.0, 1.0, -1.0, -1.0])]
    results = [minimize(negative_log_likelihood, start, method="L-BFGS-B", bounds=PARAMETER_BOUNDS) for start in starts]
    result = min(results, key=lambda item: item.fun)
    parameters = result.x
    probs = softmax(utility(parameters, sample["P_Regular"], sample["P_Premium"]))
    at_bound = {PARAMETER_NAMES[i]: bool(abs(parameters[i] - PARAMETER_BOUNDS[i][0]) < 1e-5 or abs(parameters[i] - PARAMETER_BOUNDS[i][1]) < 1e-5) for i in range(4)}
    return {"sample": sample, "params": parameters, "probs": probs, "predicted": probs.argmax(axis=1), "observed": observed, "at_bound": at_bound, "nll": float(result.fun), "success": bool(result.success), "observed_counts": sample["Choice"].value_counts().to_dict()}


MODELS = {household: fit_household(household) for household in HOUSEHOLDS}


def probabilities(model, regular_price, premium_price):
    q = softmax(utility(model["params"], [regular_price], [premium_price]))[0]
    return dict(zip(CHOICES, q.astype(float)))


def boundaries(model):
    asc_regular, asc_premium, beta_regular, beta_premium = model["params"]
    regular_no_purchase = PR_MEAN - asc_regular * PR_STD / beta_regular if abs(beta_regular) > 1e-12 else np.nan
    premium_no_purchase = PP_MEAN - asc_premium * PP_STD / beta_premium if abs(beta_premium) > 1e-12 else np.nan
    if abs(beta_premium) > 1e-12:
        slope = beta_regular * PP_STD / (beta_premium * PR_STD)
        intercept = PP_MEAN + (asc_regular - asc_premium) * PP_STD / beta_premium - slope * PR_MEAN
    else:
        slope, intercept = np.nan, np.nan
    return regular_no_purchase, premium_no_purchase, slope, intercept


def decision_map(household, regular_price, premium_price):
    model = MODELS[household]; sample = model["sample"]
    xs = np.linspace(30, 70, 180); ys = np.linspace(55, 145, 180); xx, yy = np.meshgrid(xs, ys)
    probs = softmax(utility(model["params"], xx.ravel(), yy.ravel())); region = probs.argmax(axis=1).reshape(xx.shape)
    fig = go.Figure(go.Heatmap(x=xs, y=ys, z=region, zmin=0, zmax=2, colorscale=[[0,"rgba(22,119,255,.12)"],[.33,"rgba(22,119,255,.12)"],[.34,"rgba(255,138,31,.10)"],[.66,"rgba(255,138,31,.10)"],[.67,"rgba(170,180,192,.12)"],[1,"rgba(170,180,192,.12)"]], showscale=False, hoverinfo="skip"))
    regular_no_purchase, premium_no_purchase, slope, intercept = boundaries(model)
    if np.isfinite(regular_no_purchase) and 30 <= regular_no_purchase <= 70:
        fig.add_vline(x=regular_no_purchase, line_width=2, line_dash="dash", line_color="#1677ff")
    if np.isfinite(premium_no_purchase) and 55 <= premium_no_purchase <= 145:
        fig.add_hline(y=premium_no_purchase, line_width=2, line_dash="dash", line_color="#ff8a1f")
    if np.isfinite(slope) and np.isfinite(intercept):
        line = slope * xs + intercept; mask = (line >= 55) & (line <= 145)
        if mask.any(): fig.add_trace(go.Scatter(x=xs[mask], y=line[mask], mode="lines", name="Regular = Premium", line={"color":"#172033","width":2}))
    for choice in CHOICES:
        data = sample[sample["Choice"] == choice]
        if data.empty: continue
        fig.add_trace(go.Scatter(x=data["P_Regular"], y=data["P_Premium"], mode="markers", name=choice, marker={"size":10,"color":COLORS[choice],"line":{"color":"white","width":1.5}}, customdata=data[["T","Quantity"]].to_numpy(), hovertemplate="<b>%{fullData.name}</b><br>Week %{customdata[0]}<br>Regular €%{x:.0f}<br>Premium €%{y:.0f}<br>Quantity %{customdata[1]}<extra></extra>"))
    scenario = probabilities(model, regular_price, premium_price); predicted = max(CHOICES, key=scenario.get)
    fig.add_trace(go.Scatter(x=[regular_price], y=[premium_price], mode="markers", name="Scenario", marker={"symbol":"star","size":18,"color":"#172033","line":{"color":"white","width":2}}))
    fig.update_layout(height=500, margin={"l":10,"r":10,"t":20,"b":10}, paper_bgcolor="white", plot_bgcolor="white", font={"family":"Inter, Arial, sans-serif","color":"#172033"}, xaxis={"title":"Regular price (€)","range":[30,70],"gridcolor":"#e8edf3","zeroline":False,"fixedrange":True}, yaxis={"title":"Premium price (€)","range":[55,145],"gridcolor":"#e8edf3","zeroline":False,"fixedrange":True}, legend={"orientation":"h","y":1.04,"x":0}, hoverlabel={"bgcolor":"white"})
    return fig, scenario, predicted


def quantity_stats(frame):
    regular = frame.loc[frame["Choice"] == "Regular", "Regular"]; premium = frame.loc[frame["Choice"] == "Premium", "Premium"]
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
    regular_price = st.slider("Regular price", 30, 70, BASELINE_R, key="mnl_regular_price")
with premium_col:
    premium_price = st.slider("Premium price", 65, 105, BASELINE_P, key="mnl_premium_price")

fig, scenario, predicted = decision_map(household, regular_price, premium_price)
model = MODELS[household]
avg_qty, regular_qty, premium_qty = quantity_stats(model["sample"])
rn, pn, slope, intercept = boundaries(model)
asc_r, asc_p, beta_r, beta_p = model["params"]
hits = [name for name, hit in model["at_bound"].items() if hit]
counts = model["observed_counts"]

st.markdown("""
<div class="top-shell"><div style="display:flex;align-items:center;justify-content:space-between;gap:18px;">
<div class="brand"><span class="brand-icon">☕</span> Coffee Capsule Lab</div>
<div class="nav"><a class="active" href="#home">Home</a><a href="#model">Model</a><a href="#data">Data</a><a href="#methodology">Methodology</a><a href="#insights">Insights</a></div>
<div class="profile">◦</div></div></div><div id="home"></div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero"><div class="hero-copy"><div class="eyebrow">Multinomial logit model</div><h1>Household Choice<br>Model</h1>
<p>Estimate choice probabilities and equal-utility boundaries for Regular, Premium, and No Purchase using an independent MNL for each household.</p></div>
<div class="controls"><div class="section-label">Current scenario</div><div style="color:#172033;font-size:13px;font-weight:600;">Regular €{regular_price} &nbsp;·&nbsp; Premium €{premium_price}</div>
<div class="card-sub" style="margin-top:5px;">The selected household and price pair drive every chart below.</div></div></div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Most likely choice", predicted)
m2.metric("P(Regular)", f"{scenario['Regular']:.0%}")
m3.metric("P(Premium)", f"{scenario['Premium']:.0%}")
m4.metric("P(No Purchase)", f"{scenario['No Purchase']:.0%}")

st.markdown('<div id="model"></div>', unsafe_allow_html=True)
left, right = st.columns([1.65, 0.8], gap="medium")
with left:
    st.markdown('<div class="card"><div class="card-title">Choice map</div><div class="card-sub">Observed choices, model-implied utility boundaries, and the current price scenario.</div></div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
with right:
    st.markdown(f'<div class="insight"><div class="eyebrow">Scenario result</div><h2>{predicted}</h2><p>At Regular €{regular_price} and Premium €{premium_price}</p><div style="height:12px"></div><div class="prob-row"><span>Regular</span><b>{scenario["Regular"]:.1%}</b></div><div class="prob-row"><span>Premium</span><b>{scenario["Premium"]:.1%}</b></div><div class="prob-row"><span>No Purchase</span><b>{scenario["No Purchase"]:.1%}</b></div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="card"><div class="card-title">Model snapshot</div><div class="card-sub">{household} · 11 observations</div><div class="kpi-value">{model["nll"]:.1f}</div><div class="card-sub">negative log-likelihood</div><div style="height:8px"></div><span class="pill">Regular {counts.get("Regular",0)}</span><span class="pill">Premium {counts.get("Premium",0)}</span><span class="pill">No Purchase {counts.get("No Purchase",0)}</span></div>', unsafe_allow_html=True)

st.markdown('<div id="insights"></div>', unsafe_allow_html=True)
i1, i2, i3 = st.columns(3)
with i1:
    st.markdown('<div class="card"><div class="card-title">Utility specification</div><div class="card-sub">No Purchase is the reference alternative.</div><div class="equation">Vᵣ = ASCᵣ + βᵣ · zᵣ</div><div class="equation">Vₚ = ASCₚ + βₚ · zₚ</div><div class="equation">Vₙ = 0</div></div>', unsafe_allow_html=True)
with i2:
    st.markdown(f'<div class="card"><div class="card-title">Selected household parameters</div><div class="card-sub">{household}</div><div class="kpi-value">{asc_r:.2f} / {asc_p:.2f}</div><div class="card-sub">ASC Regular / Premium</div><div class="kpi-value">{beta_r:.2f} / {beta_p:.2f}</div><div class="card-sub">Price coefficient Regular / Premium</div></div>', unsafe_allow_html=True)
with i3:
    st.markdown(f'<div class="card"><div class="card-title">Choice boundaries</div><div class="card-sub">Equal-utility thresholds in original price units.</div><div class="equation">Regular = Premium<br>Pₚ = {slope:.2f} × Pᵣ + {intercept:.1f}</div><div class="card-sub" style="margin-top:9px;">Regular = No Purchase: {rn:.1f} €</div><div class="card-sub">Premium = No Purchase: {pn:.1f} €</div></div>', unsafe_allow_html=True)

if hits:
    st.markdown(f'<div class="warning"><b>Identification warning.</b> {", ".join(hits)} reached the ±20 optimisation bound. With only 11 observations and missing choice categories, these parameters are weakly identified. Treat the associated estimates as exploratory rather than precise economic effects.</div>', unsafe_allow_html=True)

st.markdown('<div id="methodology"></div>', unsafe_allow_html=True)
st.subheader("Model & methodology")
with st.expander("How the MNL works"):
    st.markdown(f"""For **{household}**, the model is estimated independently from that household's 11 weekly choices.

The systematic utilities are:
- `V_R = ASC_R + β_R × z_R`
- `V_P = ASC_P + β_P × z_P`
- `V_N = 0`

The choice probabilities use the softmax function:
`P(i) = exp(V_i) / [exp(V_R) + exp(V_P) + exp(V_N)]`

The parameters are estimated by maximum likelihood: the optimiser searches for the parameter values that make the observed choices as probable as possible.

Price variables are standardised for numerical stability; the boundaries are transformed back into euros for the chart.
""")
with st.expander("How should I read the boundaries?"):
    st.markdown("""- **Dashed vertical line:** Regular = No Purchase.
- **Dashed horizontal line:** Premium = No Purchase.
- **Solid diagonal line:** Regular = Premium.
- **Dots:** observed household-week choices.
- **Star:** current scenario.

These are **equal-utility model boundaries**, not validated willingness-to-pay thresholds.""")
with st.expander("Why independent household MNLs?"):
    st.markdown("""The independent specification lets the dashboard estimate a distinct preference structure for each household. The trade-off is statistical power: each household has only 11 observations, and some alternatives are never observed.

This also means the model should be treated as an exploratory discrete-choice exercise rather than a validated causal demand model. Standard MNL assumptions, including IIA, should also be kept in mind.""")

st.markdown('<div id="data"></div>', unsafe_allow_html=True)
st.subheader("Observed weeks")
sample = model["sample"][["T", "P_Regular", "P_Premium", "Choice", "Quantity"]].copy()
sample.columns = ["Week", "Regular price", "Premium price", "Observed choice", "Quantity"]
st.dataframe(sample, use_container_width=True, hide_index=True)
st.caption("33 total household-week observations · 11 observations per household · independent MNL estimates")
