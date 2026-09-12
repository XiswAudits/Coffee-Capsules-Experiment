import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Coffee Capsule Demand", page_icon="☕", layout="wide")

# -----------------------------------------------------------------------------
# Data and labels
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
            if regular_q > 0:
                choice = "Regular"
            elif premium_q > 0:
                choice = "Premium"
            else:
                choice = "No Purchase"
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
# Proposed classification model
#
# The behavioral process is modeled in two stages:
#   1) Buy vs. No Purchase
#   2) Conditional on buying: Regular vs. Premium
#
# Both are logistic classifiers. Since the inputs are prices, each 50% contour is
# a straight line in the P_R / P_P plane. Household indicators allow the pooled
# experiment to account for systematic household differences.
# -----------------------------------------------------------------------------
PR_MEAN = OBS["P_Regular"].mean()
PR_STD = OBS["P_Regular"].std(ddof=0)
PP_MEAN = OBS["P_Premium"].mean()
PP_STD = OBS["P_Premium"].std(ddof=0)


def features(frame):
    return np.column_stack([
        (frame["P_Regular"].to_numpy() - PR_MEAN) / PR_STD,
        (frame["P_Premium"].to_numpy() - PP_MEAN) / PP_STD,
        (frame["Household"] == "Household 2").astype(float).to_numpy(),
        (frame["Household"] == "Household 3").astype(float).to_numpy(),
    ])


purchase_y = (OBS["Choice"] != "No Purchase").astype(int).to_numpy()
purchase_model = LogisticRegression(C=1.0, max_iter=5000).fit(features(OBS), purchase_y)

BUYERS = OBS[OBS["Choice"] != "No Purchase"].copy()
choice_y = (BUYERS["Choice"] == "Premium").astype(int).to_numpy()
choice_model = LogisticRegression(C=1.0, max_iter=5000).fit(features(BUYERS), choice_y)


def probabilities(household, p_regular, p_premium):
    x = pd.DataFrame({
        "Household": [household],
        "P_Regular": [p_regular],
        "P_Premium": [p_premium],
    })
    p_buy = purchase_model.predict_proba(features(x))[0, 1]
    p_premium_given_buy = choice_model.predict_proba(features(x))[0, 1]
    return {
        "No Purchase": 1.0 - p_buy,
        "Regular": p_buy * (1.0 - p_premium_given_buy),
        "Premium": p_buy * p_premium_given_buy,
    }


def boundary(model, household):
    """Return P_P = slope * P_R + intercept for the model's 50% contour."""
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    household_effect = 0.0
    if household == "Household 2":
        household_effect = coef[2]
    elif household == "Household 3":
        household_effect = coef[3]

    # b0 + bR*zR + bP*zP + household_effect = 0
    if abs(coef[1]) < 1e-12:
        return np.nan, np.nan
    slope_z = -coef[0] / coef[1]
    intercept_z = -(intercept + household_effect) / coef[1]
    slope = slope_z * PP_STD / PR_STD
    raw_intercept = PP_MEAN + PP_STD * intercept_z - slope * PR_MEAN
    return slope, raw_intercept


def decision_map(household, scenario_regular, scenario_premium):
    x_min, x_max = 30, 70
    y_min, y_max = 65, 105
    xs = np.linspace(x_min, x_max, 180)
    ys = np.linspace(y_min, y_max, 180)
    xx, yy = np.meshgrid(xs, ys)
    grid = pd.DataFrame({
        "Household": household,
        "P_Regular": xx.ravel(),
        "P_Premium": yy.ravel(),
    })

    p_buy = purchase_model.predict_proba(features(grid))[:, 1].reshape(xx.shape)
    p_premium = choice_model.predict_proba(features(grid))[:, 1].reshape(xx.shape)
    # 0 = Regular, 1 = Premium, 2 = No Purchase
    zones = np.where(p_buy < 0.5, 2, np.where(p_premium >= 0.5, 1, 0))

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=zones, zmin=0, zmax=2,
        colorscale=[
            [0.00, "rgba(37,99,235,0.10)"], [0.33, "rgba(37,99,235,0.10)"],
            [0.34, "rgba(216,155,0,0.10)"], [0.66, "rgba(216,155,0,0.10)"],
            [0.67, "rgba(100,116,139,0.10)"], [1.00, "rgba(100,116,139,0.10)"],
        ], showscale=False, hoverinfo="skip"
    ))

    purchase_slope, purchase_intercept = boundary(purchase_model, household)
    choice_slope, choice_intercept = boundary(choice_model, household)
    purchase_line = purchase_slope * xs + purchase_intercept
    choice_line = choice_slope * xs + choice_intercept

    purchase_mask = (purchase_line >= y_min) & (purchase_line <= y_max)
    choice_mask = (choice_line >= y_min) & (choice_line <= y_max)

    if purchase_mask.any():
        fig.add_trace(go.Scatter(
            x=xs[purchase_mask], y=purchase_line[purchase_mask],
            mode="lines", name="Purchase boundary",
            line=dict(color="#111827", width=3, dash="dash"),
            hovertemplate="Purchase boundary<br>P<sub>P</sub>=%{y:.1f}<extra></extra>",
        ))
    if choice_mask.any():
        fig.add_trace(go.Scatter(
            x=xs[choice_mask], y=choice_line[choice_mask],
            mode="lines", name="Regular / Premium boundary",
            line=dict(color="#111827", width=3),
            hovertemplate="Regular / Premium boundary<br>P<sub>P</sub>=%{y:.1f}<extra></extra>",
        ))

    household_data = OBS[OBS["Household"] == household]
    for choice in CHOICES:
        sample = household_data[household_data["Choice"] == choice]
        if sample.empty:
            continue
        # IMPORTANT: use the actual T column. DataFrame.T means transpose in pandas.
        customdata = sample[["T", "Quantity"]].to_numpy()
        fig.add_trace(go.Scatter(
            x=sample["P_Regular"], y=sample["P_Premium"],
            mode="markers", name=choice,
            marker=dict(size=12, color=POINTS[choice], line=dict(color="white", width=1.5)),
            customdata=customdata,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>Week %{customdata[0]}<br>"
                "P<sub>R</sub>=%{x:.0f}<br>P<sub>P</sub>=%{y:.0f}<br>"
                "Quantity=%{customdata[1]}<extra></extra>"
            ),
        ))

    scenario_probs = probabilities(household, scenario_regular, scenario_premium)
    predicted = max(CHOICES, key=scenario_probs.get)
    fig.add_trace(go.Scatter(
        x=[scenario_regular], y=[scenario_premium], mode="markers",
        name="Your scenario",
        marker=dict(symbol="star", size=20, color="#111827", line=dict(color="white", width=2)),
        hovertemplate=(
            f"<b>Your scenario</b><br>P<sub>R</sub>={scenario_regular:.0f}<br>"
            f"P<sub>P</sub>={scenario_premium:.0f}<br>Most likely: {predicted}<extra></extra>"
        ),
    ))

    fig.update_layout(
        height=600, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title="Regular capsule price (P_R)", range=[x_min, x_max], gridcolor="#E5E7EB", zeroline=False),
        yaxis=dict(title="Premium capsule price (P_P)", range=[y_min, y_max], gridcolor="#E5E7EB", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    return fig, scenario_probs, (purchase_slope, purchase_intercept), (choice_slope, choice_intercept)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {max-width:1320px; padding-top:1.5rem; padding-bottom:3rem;}
section[data-testid="stSidebar"] {border-right:1px solid #E7E9EE;}
.hero {background:linear-gradient(135deg,#FFFDF7 0%,#F7F8FA 100%); border:1px solid #E7E9EE;
       border-radius:24px; padding:30px 34px; margin-bottom:22px;}
.kicker {font-size:.76rem; letter-spacing:.13em; text-transform:uppercase; font-weight:700; color:#667085; margin-bottom:8px;}
.hero h1 {color:#172033; font-size:2.5rem; line-height:1.06; letter-spacing:-.045em; margin:0 0 10px;}
.hero p {color:#667085; font-size:1.03rem; margin:0; max-width:880px;}
.note {color:#667085; font-size:.92rem; margin-top:-8px;}
.callout {border-left:4px solid #172033; background:#F7F8FA; padding:14px 17px; border-radius:0 12px 12px 0;}
[data-testid="stMetric"] {border:1px solid #E7E9EE; border-radius:16px; padding:14px 16px; background:#fff;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ☕ Coffee Capsule Lab")
    st.caption("Explore how the two prices shape household choice.")
    household = st.selectbox("Household", list(HOUSEHOLDS))
    st.divider()
    st.markdown("**Try a price scenario**")
    scenario_regular = st.slider("Regular price (P_R)", 30, 70, 42)
    scenario_premium = st.slider("Premium price (P_P)", 65, 105, 93)
    st.divider()
    st.caption("The model pools all 33 household-week observations and includes household indicators.")

fig, scenario_probs, purchase_boundary, choice_boundary = decision_map(
    household, scenario_regular, scenario_premium
)
prediction = max(CHOICES, key=scenario_probs.get)

st.markdown("""
<div class="hero">
  <div class="kicker">Coffee capsule experiment</div>
  <h1>How price changes the household's choice</h1>
  <p>Change the two prices on the left. The map shows the observed choices and the model's estimated regions for <b>Regular</b>, <b>Premium</b>, and <b>No Purchase</b>.</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Most likely choice", prediction)
m2.metric("P(Regular)", f"{scenario_probs['Regular']:.0%}")
m3.metric("P(Premium)", f"{scenario_probs['Premium']:.0%}")
m4.metric("P(No purchase)", f"{scenario_probs['No Purchase']:.0%}")

st.markdown("### The decision map")
st.markdown(
    f'<div class="note">{household} · observed weekly choices + fitted decision regions · ★ = your current scenario</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

with st.expander("How to read the two boundaries"):
    st.markdown("""
**Dashed line — purchase boundary.** This is the 50% contour from the first-stage classifier. It separates the model's buying and no-purchase regions.

**Solid line — Regular / Premium boundary.** This is the 50% contour from the second-stage classifier. It is interpreted conditional on the household buying.

So the model is not forcing all three outcomes into a single household-specific line. It reflects the sequence of decisions in the experiment: **buy or not → Regular or Premium**.
""")

st.markdown("### Why this classification approach fits the proposed model")
st.markdown("""
The experiment gives us a natural classification setting: each household-week has two observable price inputs — **P_R** and **P_P** — and a behavioral outcome. A linear logistic classifier is a sensible first model because it turns those prices into probabilities while keeping the decision boundary visible on the same two-dimensional map.

The two-stage formulation is especially useful because **No Purchase is a different decision from choosing between two products**. First, a household decides whether the prices are acceptable enough to buy at all. If it buys, it then chooses between Regular and Premium. This mirrors the observed behavior much better than fitting a Regular-vs-Premium classifier and treating missing purchases as if they were another product choice.

It is also transparent. The 50% contours are straight lines, so the model can be explained directly in terms of combinations of the two prices rather than relying on a black-box prediction. At the same time, pooling the observations and adding household indicators lets us use information from the complete experiment when an individual household does not contain all possible outcomes.
""")
st.markdown(
    '<div class="callout"><b>Why Household 3 now has a No Purchase region</b><br>'
    'Household 3 has 7 Premium weeks, 4 No Purchase weeks, and 0 Regular weeks. The old app only estimated a Regular-vs-Premium classifier, so it had no model for the purchase decision. The revised first stage explicitly models Buy vs. No Purchase, which is why the graph can now show a No Purchase region. The Regular/Premium boundary is learned from the purchasing observations across the experiment; it should not be interpreted as evidence that Household 3 itself ever chose Regular.</div>',
    unsafe_allow_html=True,
)

st.markdown("### What the model says for this scenario")
b1, b2 = st.columns(2)
with b1:
    ps, pi = purchase_boundary
    st.markdown("**Purchase stage**")
    st.write(f"Estimated 50% boundary: `P_P = {ps:.3f} × P_R + {pi:.2f}`")
    st.caption("This line describes the transition between buying and not buying.")
with b2:
    cs, ci = choice_boundary
    st.markdown("**Product stage**")
    st.write(f"Estimated 50% boundary: `P_P = {cs:.3f} × P_R + {ci:.2f}`")
    st.caption("This line describes Regular vs. Premium conditional on buying.")

# -----------------------------------------------------------------------------
# Supporting sections
# -----------------------------------------------------------------------------
t1, t2, t3 = st.tabs(["Household insights", "Data & validation", "Model notes"])

with t1:
    h = OBS[OBS["Household"] == household]
    counts = h["Choice"].value_counts().reindex(CHOICES).fillna(0).astype(int)
    a, b, c = st.columns(3)
    a.metric("Regular weeks", int(counts["Regular"]))
    b.metric("Premium weeks", int(counts["Premium"]))
    c.metric("No-purchase weeks", int(counts["No Purchase"]))

    st.markdown("#### Observed behavior")
    if household == "Household 1":
        st.write("Household 1 is mostly Regular-oriented and switches to Premium in the weeks where the relative price relationship becomes more attractive. There is no observed No Purchase week, so the purchase stage necessarily borrows information from the pooled experiment.")
    elif household == "Household 2":
        st.write("Household 2 switches between Regular and Premium, with Premium appearing in four observed weeks. There is no observed No Purchase week, so the no-purchase region is identified from the pooled purchase response rather than from Household 2 alone.")
    else:
        st.write("Household 3 is the clearest illustration of why the two-stage structure helps. It has Premium purchases and No Purchase observations, but no Regular purchases. The first stage can therefore describe purchase vs. non-purchase while the second stage describes the product choice using the purchasing sample.")

    st.markdown("#### Quantity when the household buys")
    quantity = h[h["Choice"] != "No Purchase"].groupby("Choice")["Quantity"].agg(["count", "mean"]).reindex(["Regular", "Premium"]).dropna()
    st.dataframe(quantity.rename(columns={"count": "Observed weeks", "mean": "Average capsules/week"}), use_container_width=True)

with t2:
    purchase_accuracy = accuracy_score(purchase_y, purchase_model.predict(features(OBS)))
    choice_accuracy = accuracy_score(choice_y, choice_model.predict(features(BUYERS)))
    a, b = st.columns(2)
    a.metric("Purchase classifier", f"{purchase_accuracy:.0%} training accuracy")
    b.metric("Regular/Premium classifier", f"{choice_accuracy:.0%} training accuracy")
    st.caption("These are in-sample figures. With only 33 household-week observations, they should be treated as descriptive rather than proof of out-of-sample predictive performance.")

    st.markdown("#### Household observations")
    st.dataframe(
        h[["T", "P_Regular", "P_Premium", "Regular", "Premium", "Choice", "Quantity"]],
        use_container_width=True,
        hide_index=True,
    )

with t3:
    st.markdown("#### Why the old graph could not show No Purchase for Household 3")
    st.write("The previous implementation fitted a separate Regular-vs-Premium model for each household. Household 3 has no Regular observations, and that model did not contain a separate purchase decision. As a result, there was no defensible mechanism for drawing a No Purchase zone.")
    st.markdown("#### What changed")
    st.write("The revised model separates purchase incidence from product choice. The purchase classifier is estimated on all household-week observations, while the Regular/Premium classifier is estimated only among weeks in which a purchase occurred. Household indicators shift the pooled decision functions for Household 2 and Household 3.")
    st.markdown("#### What this model is — and is not")
    st.write("This is an interpretable exploratory demand-classification model. It is not a structural utility model, a causal elasticity estimate, or a validated willingness-to-pay model. The small sample is the main limitation, and stronger economic conclusions would require a larger experiment.")
    st.markdown("#### Natural next step")
    st.write("With more weeks and households, the same logic could be extended into a conditional choice model with richer household heterogeneity, out-of-sample validation, price elasticities and a separate quantity-demand component.")

st.divider()
st.caption("Coffee Capsule Demand Lab · Exploratory analysis of the supplied 11-week experiment · 33 household-week observations")
