import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Coffee Capsule Demand Classifier",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Dataset
# -----------------------------
DATA = [
    (1, 40, 70, 5, 0, 0, 3, 0, 4),
    (2, 35, 81, 5, 0, 4, 0, 0, 4),
    (3, 50, 100, 4, 0, 4, 0, 0, 0),
    (4, 65, 75, 0, 4, 0, 3, 0, 4),
    (5, 50, 70, 0, 4, 0, 3, 0, 5),
    (6, 55, 80, 4, 0, 4, 0, 0, 4),
    (7, 40, 90, 4, 0, 4, 0, 0, 0),
    (8, 45, 92, 4, 0, 4, 0, 0, 0),
    (9, 47, 91, 4, 0, 4, 0, 0, 0),
    (10, 35, 70, 5, 0, 0, 3, 0, 5),
    (11, 45, 80, 4, 0, 4, 0, 0, 4),
]
COLUMNS = [
    "T", "P_Regular", "P_Premium",
    "HH1_Regular", "HH1_Premium",
    "HH2_Regular", "HH2_Premium",
    "HH3_Regular", "HH3_Premium"
]
df = pd.DataFrame(DATA, columns=COLUMNS)

HOUSEHOLDS = {
    "Household 1": ("HH1_Regular", "HH1_Premium"),
    "Household 2": ("HH2_Regular", "HH2_Premium"),
    "Household 3": ("HH3_Regular", "HH3_Premium"),
}

CHOICES = ["Regular", "Premium", "No Purchase"]
LABEL_COLORS = {"Regular": "#2F80ED", "Premium": "#F2C94C", "No Purchase": "#8A8F98"}

# -----------------------------
# Helpers
# -----------------------------
def add_choice_columns(frame, hh):
    reg_col, prem_col = HOUSEHOLDS[hh]
    x = frame.copy()
    x["Choice"] = np.select(
        [x[reg_col] > 0, x[prem_col] > 0],
        ["Regular", "Premium"],
        default="No Purchase",
    )
    x["Quantity"] = np.where(
        x["Choice"].eq("Regular"), x[reg_col],
        np.where(x["Choice"].eq("Premium"), x[prem_col], 0)
    )
    return x

def fit_binary_model(hh):
    """Fit the requested linear classifier when exactly two choices are observed."""
    x = add_choice_columns(df, hh)
    observed = x["Choice"].unique().tolist()
    if len(observed) != 2:
        return None, x, "A binary model requires exactly two observed choices."
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10.0, max_iter=2000)
    )
    X = x[["P_Regular", "P_Premium"]]
    y = x["Choice"]
    model.fit(X, y)

    # Convert the standardized logistic equation back to raw price units.
    scaler = model.named_steps["standardscaler"]
    clf = model.named_steps["logisticregression"]
    raw_coef = clf.coef_[0] / scaler.scale_
    raw_intercept = clf.intercept_[0] - np.sum(clf.coef_[0] * scaler.mean_ / scaler.scale_)

    # sklearn's positive class is clf.classes_[1].
    # Boundary: raw_coef[0]*PR + raw_coef[1]*PP + intercept = 0
    if abs(raw_coef[1]) < 1e-12:
        m, b = np.nan, np.nan
    else:
        m = -raw_coef[0] / raw_coef[1]
        b = -raw_intercept / raw_coef[1]

    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    return {
        "model": model,
        "coef": raw_coef,
        "intercept": raw_intercept,
        "slope": m,
        "intercept_b": b,
        "accuracy": acc,
        "classes": list(clf.classes_),
    }, x, None

def fit_three_class_model():
    """Reference model across all household-week observations."""
    rows = []
    for hh, (reg_col, prem_col) in HOUSEHOLDS.items():
        x = add_choice_columns(df, hh)
        for _, r in x.iterrows():
            rows.append({
                "Household": hh,
                "T": r["T"],
                "P_Regular": r["P_Regular"],
                "P_Premium": r["P_Premium"],
                "Choice": r["Choice"],
                "Quantity": r["Quantity"],
            })
    all_obs = pd.DataFrame(rows)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=5.0, max_iter=3000, multi_class="auto")
    )
    model.fit(all_obs[["P_Regular", "P_Premium"]], all_obs["Choice"])
    all_obs["Predicted"] = model.predict(all_obs[["P_Regular", "P_Premium"]])
    return model, all_obs

def zone_for_prediction(model_info, pr, pp):
    model = model_info["model"]
    return model.predict(pd.DataFrame({"P_Regular": [pr], "P_Premium": [pp]}))[0]

def quantity_by_choice(obs):
    return obs.groupby("Choice")["Quantity"].mean().to_dict()

def make_chart(hh, model_info, obs, sim_pr, sim_pp):
    x_min, x_max = 25, 75
    y_min, y_max = 50, 110
    xs = np.linspace(x_min, x_max, 100)
    ys = np.linspace(y_min, y_max, 120)
    xx, yy = np.meshgrid(xs, ys)

    fig = go.Figure()

    # Background decision regions for the fitted binary classifier.
    if model_info is not None:
        grid = pd.DataFrame({
            "P_Regular": xx.ravel(),
            "P_Premium": yy.ravel()
        })
        labels = model_info["model"].predict(grid).reshape(xx.shape)
        class_to_num = {c: i for i, c in enumerate(model_info["classes"])}
        zz = np.vectorize(class_to_num.get)(labels)

        # Use a translucent heatmap. This is deliberately subtle so the data remains primary.
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=zz,
            colorscale=[
                [0.00, "rgba(47,128,237,0.10)"],
                [0.49, "rgba(47,128,237,0.10)"],
                [0.50, "rgba(242,201,76,0.10)"],
                [0.99, "rgba(242,201,76,0.10)"],
                [1.00, "rgba(138,143,152,0.10)"],
            ],
            showscale=False,
            hoverinfo="skip",
        ))

        m = model_info["slope"]
        b = model_info["intercept_b"]
        if np.isfinite(m) and np.isfinite(b):
            line_y = m * xs + b
            mask = (line_y >= y_min) & (line_y <= y_max)
            fig.add_trace(go.Scatter(
                x=xs[mask], y=line_y[mask],
                mode="lines",
                name="Linear decision boundary",
                line=dict(color="#111827", width=3),
                hovertemplate="Boundary<br>P<sub>P</sub> = %{y:.2f}<extra></extra>",
            ))

    for choice in CHOICES:
        sub = obs[obs["Choice"] == choice]
        if len(sub):
            fig.add_trace(go.Scatter(
                x=sub["P_Regular"], y=sub["P_Premium"],
                mode="markers",
                name=choice,
                marker=dict(
                    size=13,
                    color=LABEL_COLORS[choice],
                    line=dict(color="white", width=1.5),
                ),
                customdata=np.c_[sub["T"], sub["Quantity"]],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Week T=%{customdata[0]}<br>"
                    "P<sub>R</sub>=%{x}<br>"
                    "P<sub>P</sub>=%{y}<br>"
                    "Quantity=%{customdata[1]}<extra></extra>"
                ),
            ))

    pred = zone_for_prediction(model_info, sim_pr, sim_pp) if model_info else "Not estimable"
    fig.add_trace(go.Scatter(
        x=[sim_pr], y=[sim_pp],
        mode="markers",
        name="Simulation",
        marker=dict(symbol="star", size=20, color="#111827", line=dict(color="white", width=2)),
        hovertemplate=(
            f"<b>Simulation</b><br>P<sub>R</sub>={sim_pr}<br>"
            f"P<sub>P</sub>={sim_pp}<br>Prediction={pred}<extra></extra>"
        ),
    ))

    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Regular capsule price (P_R)",
        yaxis_title="Premium capsule price (P_P)",
        xaxis=dict(range=[x_min, x_max], gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(range=[y_min, y_max], gridcolor="rgba(0,0,0,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig

# -----------------------------
# Page styling
# -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px;}
.hero {padding: 1.2rem 1.4rem; border: 1px solid #e5e7eb; border-radius: 18px;
       background: linear-gradient(135deg,#ffffff,#f7f8fa); margin-bottom: 1rem;}
.hero h1 {margin:0 0 .35rem 0; letter-spacing:-0.03em;}
.hero p {margin:0; color:#59636e; font-size:1.05rem;}
.metric-card {border:1px solid #e5e7eb; border-radius:14px; padding:1rem; background:#fff;}
.small {color:#69727d; font-size:.9rem;}
.zone {border-left:4px solid #111827; padding:.7rem 1rem; background:#f8fafc; border-radius:0 10px 10px 0;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("☕ Demand Lab")
hh = st.sidebar.selectbox("Household", list(HOUSEHOLDS.keys()))

sim_pr = st.sidebar.slider("Simulated Regular Price (P_R)", 30, 70, 50)
sim_pp = st.sidebar.slider("Simulated Premium Price (P_P)", 60, 110, 80)

obs = add_choice_columns(df, hh)
model_info, _, model_error = fit_binary_model(hh)

st.sidebar.divider()
st.sidebar.caption("Model scope")
if model_info:
    st.sidebar.success("Binary choice model is estimable for this household.")
else:
    st.sidebar.warning("A Regular/Premium boundary is not identifiable from this household's observed choices.")

# -----------------------------
# Hero
# -----------------------------
st.markdown("""
<div class="hero">
<h1>Household Coffee Capsule Demand Classifier</h1>
<p>Explore how Regular and Premium prices map to observed household choices, quantities, and estimated decision zones.</p>
</div>
""", unsafe_allow_html=True)

if model_info:
    pred = zone_for_prediction(model_info, sim_pr, sim_pp)
    q_avgs = quantity_by_choice(obs)
    est_qty = q_avgs.get(pred, 0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Simulated choice", pred)
    c2.metric("Estimated quantity", f"{est_qty:.1f} capsules/week")
    c3.metric("Classifier accuracy", f"{model_info['accuracy']*100:.0f}%")
    if np.isfinite(model_info["slope"]):
        c4.metric("Boundary slope (m)", f"{model_info['slope']:.3f}")
    else:
        c4.metric("Boundary slope (m)", "—")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observed choices", ", ".join(obs["Choice"].unique()))
    c2.metric("Regular observations", int((obs["Choice"] == "Regular").sum()))
    c3.metric("Premium observations", int((obs["Choice"] == "Premium").sum()))
    c4.metric("No-purchase observations", int((obs["Choice"] == "No Purchase").sum()))

# -----------------------------
# Main tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Decision map", "Household insights", "Data", "How it works"
])

with tab1:
    st.subheader(f"{hh}: price-choice map")
    if model_info:
        st.plotly_chart(
            make_chart(hh, model_info, obs, sim_pr, sim_pp),
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True}
        )
        m = model_info["slope"]
        b = model_info["intercept_b"]
        st.markdown(
            f"**Estimated boundary:**  "
            f"`P_P = {m:.3f} × P_R + {b:.2f}`"
        )
        st.caption(
            "The line is the empirical linear classification boundary implied by the observed "
            "Regular/Premium choices. It is not a structural utility equation."
        )
    else:
        st.plotly_chart(
            make_chart(hh, None, obs, sim_pr, sim_pp),
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True}
        )
        st.warning(
            f"{hh} does not contain both Regular and Premium outcomes in the observed data, "
            "so a Regular-vs-Premium boundary cannot be estimated for this household alone."
        )

with tab2:
    st.subheader("What the observed behavior suggests")

    if hh == "Household 1":
        st.markdown("""
**Household 1 — Regular-oriented, but responsive to relative prices.**

- Regular is purchased in 9 of 11 observed weeks.
- Premium appears at T=4 and T=5, when Regular becomes relatively expensive and/or Premium becomes relatively attractive.
- The fitted boundary provides a compact description of the Regular→Premium switching pattern.
- Average observed quantities are approximately **4.2 Regular** capsules when Regular is chosen and **4.0 Premium** when Premium is chosen.
""")
    elif hh == "Household 2":
        st.markdown("""
**Household 2 — Mostly Regular, with Premium switching when Premium becomes cheaper.**

- Regular is purchased in 7 of 11 observed weeks.
- Premium is purchased at T=1, 4, 5 and 10.
- The observed switching pattern is particularly consistent with changes in the Premium price.
- Average observed quantities are **4.0 Regular** and **3.0 Premium** capsules per purchasing week.
""")
    else:
        st.markdown("""
**Household 3 — Premium-only when willing to buy.**

- Regular is never purchased in the observed sample.
- Premium is purchased in 7 weeks and there is no purchase in 4 weeks.
- Premium demand is observed at P_P ≤ 90, while the four no-purchase observations occur at P_P = 91, 92 or 100.
- This strongly suggests a Premium willingness-to-pay threshold in this small sample, but **90 is an observed threshold pattern, not a statistically estimated structural WTP parameter**.
- A Regular-vs-Premium boundary cannot be estimated because there are no Regular observations for this household.
""")

    st.markdown("### Observed quantity by choice")
    q = obs.groupby("Choice")["Quantity"].agg(["count", "mean", "min", "max"]).reindex(CHOICES).dropna()
    st.dataframe(q.rename(columns={"count": "Weeks", "mean": "Avg capsules/week",
                                   "min": "Min", "max": "Max"}), use_container_width=True)

    st.markdown("### Important modeling caveat")
    st.info(
        "With only 11 weeks per household, the classifier is best treated as an exploratory "
        "behavioral segmentation tool. It describes the observed sample; it should not be "
        "presented as a validated causal demand or willingness-to-pay model."
    )

with tab3:
    st.subheader("Household observations")
    show = obs[[
        "T", "P_Regular", "P_Premium",
        HOUSEHOLDS[hh][0], HOUSEHOLDS[hh][1], "Choice", "Quantity"
    ]].copy()
    st.dataframe(show, use_container_width=True, hide_index=True)

    with st.expander("Show complete source dataset"):
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Understanding the model & decision zones")

    st.markdown("""
### 1. The P_R–P_P plane

Each observation is one week. The horizontal axis is the **Regular capsule price (P_R)** and
the vertical axis is the **Premium capsule price (P_P)**.

Every household-week observation is converted into one of three behavioral outcomes:

1. **Regular** — Regular quantity > 0
2. **Premium** — Premium quantity > 0
3. **No Purchase** — both quantities = 0

The goal is to understand whether combinations of prices can separate these observed choices.
""")

    st.markdown("""
### 2. What the linear boundary means

For a household with both Regular and Premium observations, the app fits a binary logistic
classifier using only `(P_R, P_P)`.

The zero-probability boundary can be written as:

**P_P = m × P_R + b**

- **m (slope):** the empirical price trade-off implied by the classifier. A positive slope means
  a higher Regular price can be associated with a higher Premium price while remaining on the
  same choice boundary.
- **b (intercept):** the fitted baseline intercept in the Regular-vs-Premium price plane.

It is safer to interpret these as **classification parameters**, rather than automatically calling
them a marginal rate of substitution or a structural willingness-to-pay parameter.
""")

    st.markdown("""
### 3. The three behavioral zones

**Premium zone:** price combinations classified as Premium by the fitted boundary.

**Regular zone:** price combinations classified as Regular by the fitted boundary.

**No-purchase zone:** a household can also have a no-purchase state when both products exceed its
willingness to buy at the observed prices. In this dataset, this state is only observed for
Household 3.

Because Household 1 and Household 2 have no observed No Purchase weeks, their household-specific
data cannot identify a third no-purchase region. Likewise, Household 3 has no Regular weeks, so
a Regular-vs-Premium boundary cannot be identified for it.
""")

    st.markdown("""
### 4. Why this matters for a demand model

The classifier answers **"which observed choice is most consistent with this price combination?"**

The quantity layer answers **"how many capsules did this household typically buy when making
that choice?"**

Combining those two layers gives a useful exploratory demand lens:

**Prices → predicted choice → historical quantity for that choice.**

A stronger next version could add more weeks/households and estimate a proper discrete-choice
demand model (for example, multinomial/conditional logit), with household heterogeneity,
price elasticities, confidence intervals, and out-of-sample validation.
""")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Exploratory analysis based on the supplied 11-week dataset. "
    "Prices and quantities are used exactly as provided."
)
