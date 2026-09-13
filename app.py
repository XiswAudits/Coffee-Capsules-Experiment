import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Coffee Capsule Demand", page_icon="☕", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
RAW = pd.read_csv("coffee_capsules_data.csv")
HOUSEHOLDS = {
    "Household 1": ("HH1_Regular", "HH1_Premium"),
    "Household 2": ("HH2_Regular", "HH2_Premium"),
    "Household 3": ("HH3_Regular", "HH3_Premium"),
}
CHOICES = ["Regular", "Premium", "No Purchase"]
POINTS = {"Regular": "#2563EB", "Premium": "#D89B00", "No Purchase": "#64748B"}


def build_observations():
    rows = []
    for household, (regular_col, premium_col) in HOUSEHOLDS.items():
        for _, row in RAW.iterrows():
            regular_q = int(row[regular_col])
            premium_q = int(row[premium_col])
            choice = "Regular" if regular_q > 0 else "Premium" if premium_q > 0 else "No Purchase"
            rows.append({
                "Household": household,
                "T": int(row["T"]),
                "P_Regular": float(row["P_Regular"]),
                "P_Premium": float(row["P_Premium"]),
                "Regular": regular_q,
                "Premium": premium_q,
                "Choice": choice,
                "Quantity": regular_q + premium_q,
            })
    return pd.DataFrame(rows)


OBS = build_observations()

# -----------------------------------------------------------------------------
# Model: two-stage pooled logistic classification
# -----------------------------------------------------------------------------
# Stage 1: Buy vs No Purchase
# Stage 2: conditional on buying, Regular vs Premium
# Household indicators let the pooled experiment capture systematic household
# differences. Standardised price variables make the coefficients numerically
# stable while the displayed equations are converted back to original prices.
# -----------------------------------------------------------------------------
PR_MEAN, PR_STD = OBS.P_Regular.mean(), OBS.P_Regular.std(ddof=0)
PP_MEAN, PP_STD = OBS.P_Premium.mean(), OBS.P_Premium.std(ddof=0)


def features(frame):
    return np.column_stack([
        (frame.P_Regular.to_numpy() - PR_MEAN) / PR_STD,
        (frame.P_Premium.to_numpy() - PP_MEAN) / PP_STD,
        (frame.Household == "Household 2").astype(float).to_numpy(),
        (frame.Household == "Household 3").astype(float).to_numpy(),
    ])


purchase_y = (OBS.Choice != "No Purchase").astype(int).to_numpy()
purchase_model = LogisticRegression(C=1.0, max_iter=5000).fit(features(OBS), purchase_y)

BUYERS = OBS[OBS.Choice != "No Purchase"].copy()
choice_y = (BUYERS.Choice == "Premium").astype(int).to_numpy()
choice_model = LogisticRegression(C=1.0, max_iter=5000).fit(features(BUYERS), choice_y)


def probabilities(household, p_regular, p_premium):
    x = pd.DataFrame({"Household": [household], "P_Regular": [p_regular], "P_Premium": [p_premium]})
    p_buy = purchase_model.predict_proba(features(x))[0, 1]
    p_premium_given_buy = choice_model.predict_proba(features(x))[0, 1]
    return {
        "No Purchase": 1 - p_buy,
        "Regular": p_buy * (1 - p_premium_given_buy),
        "Premium": p_buy * p_premium_given_buy,
        "Buy": p_buy,
        "Premium | Buy": p_premium_given_buy,
    }


def boundary(model, household):
    """Return the 50% contour as P_P = slope * P_R + intercept."""
    coef = model.coef_[0]
    household_effect = coef[2] if household == "Household 2" else coef[3] if household == "Household 3" else 0.0
    if abs(coef[1]) < 1e-12:
        return np.nan, np.nan
    slope_z = -coef[0] / coef[1]
    intercept_z = -(model.intercept_[0] + household_effect) / coef[1]
    slope = slope_z * PP_STD / PR_STD
    intercept = PP_MEAN + PP_STD * intercept_z - slope * PR_MEAN
    return float(slope), float(intercept)


def equation(slope, intercept):
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return "not estimable"
    sign = "+" if intercept >= 0 else "−"
    return f"P_P = {slope:.2f} × P_R {sign} {abs(intercept):.1f}"


def decision_map(household, scenario_regular, scenario_premium):
    p_slope, p_intercept = boundary(purchase_model, household)
    c_slope, c_intercept = boundary(choice_model, household)

    x_min, x_max = 30, 70
    candidates = [65, 105]
    for slope, intercept in [(p_slope, p_intercept), (c_slope, c_intercept)]:
        if np.isfinite(slope):
            candidates.extend([slope * x_min + intercept, slope * x_max + intercept])
    y_min = max(55, np.floor(min(candidates) - 5))
    y_max = min(145, np.ceil(max(candidates) + 5))
    if y_max - y_min < 35:
        y_max = y_min + 35

    xs = np.linspace(x_min, x_max, 190)
    ys = np.linspace(y_min, y_max, 190)
    xx, yy = np.meshgrid(xs, ys)
    grid = pd.DataFrame({"Household": household, "P_Regular": xx.ravel(), "P_Premium": yy.ravel()})
    p_buy = purchase_model.predict_proba(features(grid))[:, 1].reshape(xx.shape)
    p_premium = choice_model.predict_proba(features(grid))[:, 1].reshape(xx.shape)
    zones = np.where(p_buy < 0.5, 2, np.where(p_premium >= 0.5, 1, 0))

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=zones, zmin=0, zmax=2,
        colorscale=[
            [0.00, "rgba(37,99,235,0.10)"], [0.33, "rgba(37,99,235,0.10)"],
            [0.34, "rgba(216,155,0,0.10)"], [0.66, "rgba(216,155,0,0.10)"],
            [0.67, "rgba(100,116,139,0.10)"], [1.00, "rgba(100,116,139,0.10)"],
        ], showscale=False, hoverinfo="skip", name="Predicted regions"
    ))

    purchase_line = p_slope * xs + p_intercept if np.isfinite(p_slope) else np.full_like(xs, np.nan)
    choice_line = c_slope * xs + c_intercept if np.isfinite(c_slope) else np.full_like(xs, np.nan)
    for line, name, dash in [(purchase_line, "Buy / No Purchase", "dash"), (choice_line, "Regular / Premium", "solid")]:
        mask = np.isfinite(line) & (line >= y_min) & (line <= y_max)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=xs[mask], y=line[mask], mode="lines", name=name,
                line=dict(color="#172033", width=3, dash=dash),
                hovertemplate=f"<b>{name}</b><br>P<sub>P</sub>=%{{y:.1f}}<extra></extra>",
            ))

    household_data = OBS[OBS.Household == household]
    for choice in CHOICES:
        sample = household_data[household_data.Choice == choice]
        if sample.empty:
            continue
        customdata = sample[["T", "Quantity"]].to_numpy()
        fig.add_trace(go.Scatter(
            x=sample.P_Regular, y=sample.P_Premium, mode="markers", name=choice,
            marker=dict(size=12, color=POINTS[choice], line=dict(color="white", width=1.5)),
            customdata=customdata,
            hovertemplate=("<b>%{fullData.name}</b><br>Week %{customdata[0]}<br>"
                           "P<sub>R</sub>=%{x:.0f}<br>P<sub>P</sub>=%{y:.0f}<br>"
                           "Quantity=%{customdata[1]}<extra></extra>"),
        ))

    scenario = probabilities(household, scenario_regular, scenario_premium)
    predicted = max(CHOICES, key=scenario.get)
    fig.add_trace(go.Scatter(
        x=[scenario_regular], y=[scenario_premium], mode="markers", name="Your scenario",
        marker=dict(symbol="star", size=20, color="#172033", line=dict(color="white", width=2)),
        hovertemplate=(f"<b>Your scenario</b><br>P<sub>R</sub>={scenario_regular:.0f}<br>"
                       f"P<sub>P</sub>={scenario_premium:.0f}<br>Most likely: {predicted}<extra></extra>"),
    ))

    fig.update_layout(
        height=590, margin=dict(l=10, r=10, t=18, b=10), paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title="Regular price (P_R)", range=[x_min, x_max], gridcolor="#E5E7EB", zeroline=False),
        yaxis=dict(title="Premium price (P_P)", range=[y_min, y_max], gridcolor="#E5E7EB", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0), hoverlabel=dict(bgcolor="white"),
    )
    return fig, scenario, (p_slope, p_intercept), (c_slope, c_intercept)


# -----------------------------------------------------------------------------
# UI styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {max-width:1320px;padding-top:1.4rem;padding-bottom:3rem;}
section[data-testid="stSidebar"] {border-right:1px solid #E7E9EE;background:#FBFBFC;}
.hero {background:linear-gradient(135deg,#FFFDF7 0%,#F7F8FA 100%);border:1px solid #E7E9EE;border-radius:24px;padding:30px 34px;margin-bottom:20px;}
.kicker {font-size:.75rem;letter-spacing:.13em;text-transform:uppercase;font-weight:700;color:#667085;margin-bottom:8px;}
.hero h1 {color:#172033;font-size:2.5rem;line-height:1.05;letter-spacing:-.045em;margin:0 0 10px;}
.hero p {color:#667085;font-size:1.02rem;margin:0;max-width:900px;}
.section-intro {color:#667085;font-size:.93rem;margin-top:-8px;margin-bottom:16px;}
.info-card {height:100%;border:1px solid #E7E9EE;border-radius:18px;background:#FFFFFF;padding:19px 20px;box-shadow:0 2px 8px rgba(23,32,51,.03);}
.info-card .label {font-size:.73rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#667085;margin-bottom:7px;}
.info-card h4 {margin:0 0 8px;color:#172033;font-size:1.02rem;}
.info-card p {margin:0;color:#475467;font-size:.9rem;line-height:1.5;}
.info-card ul {margin:7px 0 0 18px;padding:0;color:#475467;font-size:.88rem;line-height:1.55;}
.insight {border:1px solid #E7E9EE;border-radius:18px;background:#fff;padding:18px 20px;height:100%;}
.insight h4 {margin:0 0 8px;color:#172033;}
.insight p {color:#475467;font-size:.9rem;line-height:1.5;margin:0;}
.callout {border-left:4px solid #172033;background:#F7F8FA;padding:14px 17px;border-radius:0 12px 12px 0;color:#344054;}
.small-note {font-size:.82rem;color:#667085;}
[data-testid="stMetric"] {border:1px solid #E7E9EE;border-radius:16px;padding:13px 15px;background:#fff;}
div[data-testid="stExpander"] {border:1px solid #E7E9EE;border-radius:14px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    title_col, info_col = st.columns([0.84, 0.16], vertical_alignment="center")
    with title_col:
        st.markdown("### ☕ Coffee Capsule Lab")
    with info_col:
        with st.popover("ⓘ"):
            st.markdown("### Model at a glance")
            st.caption("A short guide to what is being measured, how the lines are constructed, and how to interpret the results.")

            st.markdown("**Model — two-stage logistic classification**")
            st.markdown("Two pooled **binary logistic regression** classifiers are used rather than one forced three-class model.")
            st.markdown("- **Stage 1:** Buy vs. No Purchase\n- **Stage 2:** Regular vs. Premium, given Buy")

            st.markdown("**What it measures — choice probabilities**")
            st.markdown("The inputs are **P_R**, **P_P**, and household identity. The output is a probability for each of the three observed outcomes.")
            st.markdown("- P(No Purchase)\n- P(Regular)\n- P(Premium)")

            st.markdown("**The boundaries — 50% probability contours**")
            st.markdown("Each line is where its binary classifier reaches **50%**. Because the model is linear in price, the contour is a straight line.")
            st.markdown("- **Dashed:** Buy ↔ No Purchase\n- **Solid:** Regular ↔ Premium | Buy")

            st.markdown("**Important caveat — useful, but exploratory**")
            st.markdown("The sample is only **33 household-weeks**. The model is interpretable and useful for the experiment, but it is not a structural utility, causal elasticity, or validated WTP model.")

    st.caption("A simple way to explore the experiment's price-choice model.")
    household = st.selectbox("Household", list(HOUSEHOLDS))
    st.divider()
    st.markdown("**Try a price scenario**")
    scenario_regular = st.slider("Regular price (P_R)", 30, 70, 42)
    scenario_premium = st.slider("Premium price (P_P)", 65, 105, 93)
    st.divider()
    st.markdown("**Data at a glance**")
    st.caption("33 household-week observations · 11 weeks · 3 households")

fig, scenario, purchase_boundary, choice_boundary = decision_map(household, scenario_regular, scenario_premium)
prediction = max(CHOICES, key=scenario.get)

# -----------------------------------------------------------------------------
# Header and scenario
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="kicker">Coffee capsule experiment</div>
  <h1>How price changes household choice</h1>
  <p>Explore the relationship between Regular and Premium prices, observed choices, and the model's estimated <b>Buy / No Purchase</b> and <b>Regular / Premium</b> decision boundaries.</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Most likely choice", prediction)
m2.metric("P(Regular)", f"{scenario['Regular']:.0%}")
m3.metric("P(Premium)", f"{scenario['Premium']:.0%}")
m4.metric("P(No purchase)", f"{scenario['No Purchase']:.0%}")

# -----------------------------------------------------------------------------
# Decision map
# -----------------------------------------------------------------------------
st.markdown("### Decision map")
st.markdown(f'<div class="section-intro">{household} · observed weekly choices + fitted regions · ★ marks your current scenario</div>', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

with st.expander("How should I read this map?"):
    st.markdown("""
**1 · Start with the dashed line — Buy vs. No Purchase.** Above it, the model's estimated probability of buying is below 50%; below it, buying is more likely than not.

**2 · Then look at the solid line — Regular vs. Premium.** This only applies inside the buying region. One side makes Regular more likely; the other makes Premium more likely.

**3 · Read the dots separately.** Dots are actual observations from the experiment. If a No Purchase dot appears inside a predicted buying region, that is a model classification error — not a data error.

The model therefore represents the sequence **buy or not → product choice**, rather than pretending that all three outcomes are observed equally for every household.
""")

# -----------------------------------------------------------------------------
# Equations and household implications
# -----------------------------------------------------------------------------
st.markdown("### What do the equations mean?")
p_s, p_i = purchase_boundary
c_s, c_i = choice_boundary

e1, e2 = st.columns(2)
with e1:
    st.markdown(f"""
    <div class="insight">
      <h4>① Purchase equation</h4>
      <p><code>{equation(p_s, p_i)}</code></p>
      <p style="margin-top:8px">This is the <b>50% Buy / No Purchase boundary</b> for {household}. It does not say that the household pays exactly that price; it describes the price combinations where the first-stage classifier is indifferent between its two predicted classes.</p>
    </div>
    """, unsafe_allow_html=True)
with e2:
    st.markdown(f"""
    <div class="insight">
      <h4>② Product-choice equation</h4>
      <p><code>{equation(c_s, c_i)}</code></p>
      <p style="margin-top:8px">This is the <b>50% Regular / Premium boundary conditional on buying</b>. Its slope describes the price trade-off in the classifier, not a structural marginal willingness-to-pay estimate.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### What is important about the current model?")
st.markdown("""
<div class="callout">
<b>The model uses information across households because the individual household data are incomplete.</b><br><br>
Household 1 and Household 2 never record a No Purchase week, while Household 3 never records a Regular purchase. That means no household has enough variation to independently identify every part of the three-outcome problem. The pooled two-stage model is therefore a pragmatic way to use the full experiment without inventing observations that do not exist.
<br><br>
The consequence is important: a household-specific boundary should be read as a <b>model-implied boundary informed by the pooled experiment and the household effect</b>, not as a directly observed willingness-to-pay threshold for that household.
</div>
""", unsafe_allow_html=True)

h1, h2, h3 = st.columns(3)
with h1:
    st.markdown("""
    <div class="insight">
      <h4>Household 1</h4>
      <p><b>Observed:</b> Regular and Premium, but no No Purchase weeks. Its purchase boundary is therefore informed by the pooled purchase stage. The Regular/Premium behavior is directly observed and is the stronger household-specific signal.</p>
    </div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown("""
    <div class="insight">
      <h4>Household 2</h4>
      <p><b>Observed:</b> Regular and Premium, but no No Purchase weeks. If its purchase boundary sits outside the visible experimental price range, that means the estimated 50% threshold is not reached within the plotted prices — not that Household 2 can never choose No Purchase.</p>
    </div>
    """, unsafe_allow_html=True)
with h3:
    st.markdown("""
    <div class="insight">
      <h4>Household 3</h4>
      <p><b>Observed:</b> Premium and No Purchase, but no Regular. Its No Purchase region is now explicitly modeled in Stage 1. The Regular/Premium boundary is pooled evidence and should not be interpreted as Household 3 having revealed a Regular preference.</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Scenario and data details
# -----------------------------------------------------------------------------
t1, t2, t3 = st.tabs(["Scenario", "Observed data", "Model notes"])
with t1:
    st.markdown("#### Your current scenario")
    s1, s2, s3 = st.columns(3)
    s1.metric("Regular price", f"{scenario_regular:.0f}")
    s2.metric("Premium price", f"{scenario_premium:.0f}")
    s3.metric("Predicted outcome", prediction)
    st.write(f"The model estimates **{scenario['Buy']:.0%}** probability of buying something. Conditional on buying, Premium has probability **{scenario['Premium | Buy']:.0%}**.")

with t2:
    household_data = OBS[OBS.Household == household]
    st.dataframe(household_data[["T","P_Regular","P_Premium","Regular","Premium","Choice","Quantity"]], use_container_width=True, hide_index=True)
    st.caption("Prices and quantities are shown exactly as supplied in the experiment dataset.")

with t3:
    purchase_acc = accuracy_score(purchase_y, purchase_model.predict(features(OBS)))
    choice_acc = accuracy_score(choice_y, choice_model.predict(features(BUYERS)))
    a, b = st.columns(2)
    a.metric("Purchase classifier", f"{purchase_acc:.0%} in-sample accuracy")
    b.metric("Product classifier", f"{choice_acc:.0%} in-sample accuracy")
    st.markdown("#### What this model is")
    st.write("An interpretable exploratory demand-classification model using regularized logistic regression. Logistic regression estimates class probabilities and a linear decision function; the 50% contour is the point at which the binary classifier changes its predicted class.")
    st.markdown("#### What it is not")
    st.write("It is not a structural utility model, causal price-elasticity estimate, or validated willingness-to-pay measure. The small sample means the boundaries should be presented as empirical, model-implied classification boundaries.")
    st.markdown("#### Natural next step")
    st.write("With more households and weeks, the model could be extended toward a richer discrete-choice specification with household heterogeneity, out-of-sample validation, and a separate quantity-demand component.")

# -----------------------------------------------------------------------------
# Footer model explainer
# -----------------------------------------------------------------------------
st.divider()
footer_left, footer_right = st.columns([0.72, 0.28], vertical_alignment="center")
with footer_left:
    st.caption("Coffee Capsule Demand Lab · 33 household-week observations · two-stage logistic classification · exploratory analysis")
with footer_right:
    with st.popover("ⓘ More info"):
        st.markdown("### Why is this a classification model?")
        st.markdown("The model is a **classification algorithm** because the outcome being predicted is categorical: **No Purchase, Regular, or Premium**. It is not trying to predict a continuous quantity directly.")
        st.markdown("The two logistic regressions turn the price inputs (**P_R**, **P_P**) and household identity into class probabilities. A binary logistic classifier estimates the probability of one class and uses a threshold — here, **50%** — to define the predicted class.")
        st.markdown("That is also why the decision map has straight boundaries: with a linear logistic model, the 50% contour corresponds to a linear decision boundary in price space. Scikit-learn explicitly describes `LogisticRegression` as a classifier and provides `predict_proba()` for class probabilities.")
        st.markdown("**In short:** the model answers *which outcome is more likely at these prices?* It does **not** directly estimate a structural utility function, causal price elasticity, or validated willingness-to-pay.")
