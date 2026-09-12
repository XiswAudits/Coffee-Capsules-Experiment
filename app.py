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
DATA_PATH = "coffee_capsules_data.csv"
RAW = pd.read_csv(DATA_PATH)
HOUSEHOLDS = {
    "Household 1": ("HH1_Regular", "HH1_Premium"),
    "Household 2": ("HH2_Regular", "HH2_Premium"),
    "Household 3": ("HH3_Regular", "HH3_Premium"),
}
CHOICES = ["Regular", "Premium", "No Purchase"]
POINTS = {"Regular": "#2563EB", "Premium": "#D89B00", "No Purchase": "#64748B"}


def build_observations():
    rows = []
    for hh, (reg_col, prem_col) in HOUSEHOLDS.items():
        for _, r in RAW.iterrows():
            reg_q, prem_q = int(r[reg_col]), int(r[prem_col])
            choice = "Regular" if reg_q > 0 else "Premium" if prem_q > 0 else "No Purchase"
            rows.append({
                "Household": hh,
                "T": int(r["T"]),
                "P_Regular": float(r["P_Regular"]),
                "P_Premium": float(r["P_Premium"]),
                "Regular": reg_q,
                "Premium": prem_q,
                "Choice": choice,
                "Quantity": reg_q + prem_q,
            })
    return pd.DataFrame(rows)


OBS = build_observations()

# -----------------------------------------------------------------------------
# Two-stage linear classification model
# -----------------------------------------------------------------------------
# Stage 1: purchase vs. no purchase.
# Stage 2: Regular vs. Premium, conditional on purchasing.
# Both are logistic classifiers, so their boundaries are linear in (P_R, P_P).
# Household indicators allow the pooled model to retain household heterogeneity.

PR_MEAN, PR_STD = OBS.P_Regular.mean(), OBS.P_Regular.std(ddof=0)
PP_MEAN, PP_STD = OBS.P_Premium.mean(), OBS.P_Premium.std(ddof=0)


def features(frame):
    return np.column_stack([
        (frame.P_Regular.values - PR_MEAN) / PR_STD,
        (frame.P_Premium.values - PP_MEAN) / PP_STD,
        (frame.Household == "Household 2").astype(float),
        (frame.Household == "Household 3").astype(float),
    ])


PURCHASE_Y = (OBS.Choice != "No Purchase").astype(int).values
PURCHASE_MODEL = LogisticRegression(C=1.0, max_iter=5000).fit(features(OBS), PURCHASE_Y)

BUYERS = OBS[OBS.Choice != "No Purchase"].copy()
CHOICE_Y = (BUYERS.Choice == "Premium").astype(int).values
CHOICE_MODEL = LogisticRegression(C=1.0, max_iter=5000).fit(features(BUYERS), CHOICE_Y)

OBS["Purchase prediction"] = np.where(PURCHASE_MODEL.predict(features(OBS)) == 1, "Buy", "No Purchase")
OBS["Choice prediction"] = np.where(CHOICE_MODEL.predict(features(OBS)) == 1, "Premium", "Regular")
OBS["Model prediction"] = np.where(OBS["Purchase prediction"] == "No Purchase", "No Purchase", OBS["Choice prediction"])


def household_frame(hh):
    return OBS[OBS.Household == hh].copy()


def probs(hh, pr, pp):
    x = pd.DataFrame({"Household": [hh], "P_Regular": [pr], "P_Premium": [pp]})
    purchase_p = PURCHASE_MODEL.predict_proba(features(x))[0, 1]
    premium_p = CHOICE_MODEL.predict_proba(features(x))[0, 1]
    # Joint three-state probabilities implied by the two-stage structure.
    return {
        "No Purchase": 1 - purchase_p,
        "Regular": purchase_p * (1 - premium_p),
        "Premium": purchase_p * premium_p,
        "purchase": purchase_p,
        "premium_given_purchase": premium_p,
    }


def line_for(model, hh):
    # Model: sigmoid(b0 + bR*zR + bP*zP + bHH2*HH2 + bHH3*HH3).
    b = model.coef_[0]
    intercept = model.intercept_[0]
    hh_effect = b[2] if hh == "Household 2" else b[3] if hh == "Household 3" else 0.0
    # zP = slope_z * zR + intercept_z
    slope_z = -b[0] / b[1]
    intercept_z = -(intercept + hh_effect) / b[1]
    # PP = slope_raw * PR + intercept_raw
    slope_raw = slope_z * PP_STD / PR_STD
    intercept_raw = PP_MEAN + PP_STD * intercept_z - slope_raw * PR_MEAN
    return slope_raw, intercept_raw


def decision_map(hh, sim_pr, sim_pp):
    xlo, xhi = 30, 70
    ylo, yhi = 65, 105
    xs = np.linspace(xlo, xhi, 180)
    ys = np.linspace(ylo, yhi, 180)
    xx, yy = np.meshgrid(xs, ys)
    grid = pd.DataFrame({"Household": hh, "P_Regular": xx.ravel(), "P_Premium": yy.ravel()})
    purchase_prob = PURCHASE_MODEL.predict_proba(features(grid))[:, 1].reshape(xx.shape)
    premium_prob = CHOICE_MODEL.predict_proba(features(grid))[:, 1].reshape(xx.shape)
    # Encode final choice: 0 Regular, 1 Premium, 2 No Purchase.
    zone = np.where(purchase_prob < 0.5, 2, np.where(premium_prob >= 0.5, 1, 0))

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=zone,
        zmin=0, zmax=2,
        colorscale=[
            [0.00, "rgba(37,99,235,0.09)"], [0.33, "rgba(37,99,235,0.09)"],
            [0.34, "rgba(216,155,0,0.10)"], [0.66, "rgba(216,155,0,0.10)"],
            [0.67, "rgba(100,116,139,0.10)"], [1.00, "rgba(100,116,139,0.10)"],
        ], showscale=False, hoverinfo="skip", name="Decision zones"
    ))

    pm, pb = line_for(PURCHASE_MODEL, hh)
    cm, cb = line_for(CHOICE_MODEL, hh)
    purchase_y = pm * xs + pb
    choice_y = cm * xs + cb
    mask_p = (purchase_y >= ylo) & (purchase_y <= yhi)
    mask_c = (choice_y >= ylo) & (choice_y <= yhi)

    fig.add_trace(go.Scatter(
        x=xs[mask_p], y=purchase_y[mask_p], mode="lines", name="Purchase boundary",
        line=dict(color="#111827", width=3, dash="dash"),
        hovertemplate="Purchase boundary<br>P<sub>P</sub>=%{y:.1f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=xs[mask_c], y=choice_y[mask_c], mode="lines", name="Regular / Premium boundary",
        line=dict(color="#111827", width=3),
        hovertemplate="Regular / Premium boundary<br>P<sub>P</sub>=%{y:.1f}<extra></extra>"
    ))

    h = household_frame(hh)
    for choice in CHOICES:
        s = h[h.Choice == choice]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.P_Regular, y=s.P_Premium, mode="markers", name=choice,
            marker=dict(size=12, color=POINTS[choice], line=dict(color="white", width=1.5)),
            customdata=np.c_[s.T, s.Quantity],
            hovertemplate=("<b>%{fullData.name}</b><br>Week %{customdata[0]}<br>"
                           "P<sub>R</sub>=%{x:.0f}<br>P<sub>P</sub>=%{y:.0f}<br>"
                           "Quantity=%{customdata[1]}<extra></extra>"),
        ))

    p = probs(hh, sim_pr, sim_pp)
    final = max(CHOICES, key=lambda k: p[k])
    fig.add_trace(go.Scatter(
        x=[sim_pr], y=[sim_pp], mode="markers", name="Your scenario",
        marker=dict(symbol="star", size=19, color="#111827", line=dict(color="white", width=2)),
        hovertemplate=(f"<b>Your scenario</b><br>P<sub>R</sub>={sim_pr:.0f}<br>"
                       f"P<sub>P</sub>={sim_pp:.0f}<br>Most likely: {final}<extra></extra>")
    ))
    fig.update_layout(
        height=590, margin=dict(l=15, r=15, t=15, b=15),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title="Regular price (P_R)", range=[xlo, xhi], gridcolor="#E5E7EB", zeroline=False),
        yaxis=dict(title="Premium price (P_P)", range=[ylo, yhi], gridcolor="#E5E7EB", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    return fig, p, (pm, pb), (cm, cb)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.markdown("""
<style>
:root { --ink:#172033; --muted:#667085; --line:#E7E9EE; --paper:#FFFFFF; --soft:#F7F8FA; }
.block-container { max-width: 1320px; padding-top: 1.6rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] { border-right: 1px solid #E7E9EE; }
.hero { background: linear-gradient(135deg,#FFFDF7 0%,#F7F8FA 100%); border:1px solid #E7E9EE;
        border-radius:24px; padding:30px 34px; margin-bottom:22px; }
.kicker { font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; color:#667085; margin-bottom:7px; }
.hero h1 { color:#172033; font-size:2.55rem; line-height:1.05; letter-spacing:-.045em; margin:0 0 10px 0; }
.hero p { color:#667085; font-size:1.03rem; margin:0; max-width:850px; }
.section-note { color:#667085; font-size:.93rem; margin-top:-8px; }
.callout { border-left:4px solid #172033; background:#F7F8FA; padding:14px 17px; border-radius:0 12px 12px 0; }
[data-testid="stMetric"] { border:1px solid #E7E9EE; border-radius:16px; padding:14px 16px; background:#fff; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ☕ Coffee Capsule Lab")
    st.caption("A small choice experiment, made easier to explore.")
    hh = st.selectbox("Household", list(HOUSEHOLDS))
    st.divider()
    st.markdown("**Try a price scenario**")
    sim_pr = st.slider("Regular price (P_R)", 30, 70, 42)
    sim_pp = st.slider("Premium price (P_P)", 65, 105, 93)
    st.divider()
    st.caption("The model uses all 33 household-week observations. Household indicators are included so the pooled classifier can capture systematic differences between households.")

fig, p, purchase_line, choice_line = decision_map(hh, sim_pr, sim_pp)
pred = max(CHOICES, key=lambda k: p[k])

st.markdown("""
<div class="hero">
  <div class="kicker">Coffee capsule experiment</div>
  <h1>How price changes the household's choice</h1>
  <p>Move the prices on the left and see how the model separates <b>buying</b> from <b>not buying</b>, then Regular from Premium.</p>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Most likely choice", pred)
c2.metric("P(Regular)", f"{p['Regular']:.0%}")
c3.metric("P(Premium)", f"{p['Premium']:.0%}")
c4.metric("P(No purchase)", f"{p['No Purchase']:.0%}")

st.markdown("### The decision map")
st.markdown(f'<div class="section-note">{hh} · observed weekly choices + fitted decision regions · the star is your current price scenario</div>', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

with st.expander("What are the two lines?", expanded=False):
    st.markdown("""
**Dashed line — purchase boundary.** This is the 50% contour of the purchase classifier. Crossing it changes the model's most likely outcome between buying and no purchase.

**Solid line — Regular / Premium boundary.** This is the 50% contour of the product-choice classifier and is applied only after the household is predicted to buy.

That gives the model the three behavioral outcomes without pretending that Regular, Premium and No Purchase are three equally observed alternatives inside every household.
""")

st.markdown("### Why this is a good fit for the model we propose")
st.markdown("""
The experiment is fundamentally a **classification problem in a two-price space**. For every household-week we observe two explanatory variables — the Regular price and the Premium price — and an observed behavioral outcome.

A **linear logistic classifier** is a useful first model here because its decision boundary is easy to see and explain: a combination of the two prices maps to a probability of purchase or a probability of choosing Premium conditional on purchase. The 50% probability contour is a straight line in the price plane. That makes the economics intuitive without hiding the model behind a black box.

The important detail is the **two-stage structure**:

1. **Purchase decision:** buy coffee capsules vs. no purchase.
2. **Product decision:** conditional on buying, Regular vs. Premium.

This matches the behavior we actually observe. It also solves the issue that the original household-by-household binary classifier could not show a meaningful no-purchase region for Household 1 or 2, or a Regular/Premium boundary for Household 3.
""")

st.markdown('<div class="callout"><b>Why not just fit one three-class line?</b><br>The data are sparse and unbalanced at the household level. Household 3 never buys Regular, while Household 1 and 2 never record a no-purchase week. A single household-specific three-class model would therefore be asking the data to identify choices that were never observed. The pooled two-stage model uses the full experiment while keeping the interpretation straightforward.</div>', unsafe_allow_html=True)

st.markdown("### What the model is saying")
cc1,cc2 = st.columns(2)
with cc1:
    st.markdown("**Purchase stage**")
    st.write(f"For {hh}, the estimated purchase boundary is approximately `P_P = {purchase_line[0]:.3f} × P_R + {purchase_line[1]:.2f}`.")
    st.caption("Crossing this line changes the model's most likely outcome between purchase and no purchase.")
with cc2:
    st.markdown("**Product stage**")
    st.write(f"Conditional on buying, the Regular/Premium boundary is approximately `P_P = {choice_line[0]:.3f} × P_R + {choice_line[1]:.2f}`.")
    st.caption("This line only has a behavioral meaning inside the purchase region.")

# -----------------------------------------------------------------------------
# Insights
# -----------------------------------------------------------------------------
t1,t2,t3 = st.tabs(["Household insights", "Data & validation", "Model notes"])
with t1:
    h = household_frame(hh)
    counts = h.Choice.value_counts().reindex(CHOICES).fillna(0).astype(int)
    a,b,c = st.columns(3)
    a.metric("Regular weeks", counts["Regular"])
    b.metric("Premium weeks", counts["Premium"])
    c.metric("No-purchase weeks", counts["No Purchase"])
    st.markdown("#### Observed behavior")
    if hh == "Household 1":
        st.write("Household 1 is mostly Regular-oriented, with Premium appearing when the relative price relationship becomes more attractive. There is no observed no-purchase week, so the purchase boundary is learned from the pooled experiment rather than from Household 1 alone.")
    elif hh == "Household 2":
        st.write("Household 2 switches between Regular and Premium, but likewise has no observed no-purchase week. The model therefore uses the pooled purchase response to draw the no-purchase region and household-specific effects to shift it.")
    else:
        st.write("Household 3 is the key reason the two-stage model is useful: it has Premium purchases and no-purchase observations, but no Regular purchases. The graph can still show a no-purchase region, while the Regular/Premium comparison is learned from purchasing households and should not be read as directly observed Regular behavior for Household 3.")
    st.markdown("#### Quantity when the household buys")
    qty = h[h.Choice != "No Purchase"].groupby("Choice").Quantity.agg(["count","mean"]).reindex(["Regular","Premium"]).dropna()
    st.dataframe(qty.rename(columns={"count":"Observed weeks","mean":"Average capsules/week"}), use_container_width=True)

with t2:
    st.markdown("#### Model performance on the observed sample")
    purchase_acc = accuracy_score(PURCHASE_Y, PURCHASE_MODEL.predict(features(OBS)))
    choice_acc = accuracy_score(CHOICE_Y, CHOICE_MODEL.predict(features(BUYERS)))
    a,b = st.columns(2)
    a.metric("Purchase classifier", f"{purchase_acc:.0%} training accuracy")
    b.metric("Regular/Premium classifier", f"{choice_acc:.0%} training accuracy")
    st.caption("These are in-sample figures. With only 33 household-week observations, they are descriptive rather than evidence of out-of-sample predictive performance.")
    st.markdown("#### Observations")
    st.dataframe(household_frame(hh)[["T","P_Regular","P_Premium","Regular","Premium","Choice","Quantity"]], use_container_width=True, hide_index=True)

with t3:
    st.markdown("#### Why Household 3 now has a no-purchase region")
    st.write("Previously, the app fitted a Regular-vs-Premium classifier separately inside each household. That approach could not create a no-purchase zone for Household 3 because it did not model purchase versus non-purchase as a separate decision. It also could not estimate a Regular-vs-Premium boundary for Household 3 because Regular is never observed there.")
    st.write("The revised model separates those questions. The first classifier estimates whether the household buys at all. The second classifier asks which capsule type is chosen, conditional on buying. The resulting decision map can therefore display all three behavioral outcomes while remaining honest about where the evidence comes from.")
    st.markdown("#### What this model is — and is not")
    st.write("It is an interpretable exploratory demand-classification model. It is not a structural utility model, a causal price-elasticity estimate, or a validated willingness-to-pay model. More weeks and more households would be needed before making stronger economic claims.")
    st.markdown("#### Next empirical step")
    st.write("With a larger experiment, the natural extension would be to estimate a conditional/multinomial choice model with household heterogeneity, then add quantity demand and out-of-sample validation. The current app is deliberately simpler so the proposed classification logic is visible rather than buried.")

st.divider()
st.caption("Coffee Capsule Demand Lab · Exploratory model based on the supplied 11-week experiment · 33 household-week observations")
