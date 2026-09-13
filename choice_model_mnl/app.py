import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(
    page_title="Coffee Capsule Choice Model",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
COLORS = {"Regular": "#2563EB", "Premium": "#D89B00", "No Purchase": "#64748B"}


def build_observations():
    rows = []
    for household, (regular_col, premium_col) in HOUSEHOLDS.items():
        for _, row in RAW.iterrows():
            regular_q = int(row[regular_col])
            premium_q = int(row[premium_col])
            choice = "Regular" if regular_q > 0 else "Premium" if premium_q > 0 else "No Purchase"
            rows.append(
                {
                    "Household": household,
                    "T": int(row["T"]),
                    "P_Regular": float(row["P_Regular"]),
                    "P_Premium": float(row["P_Premium"]),
                    "Regular": regular_q,
                    "Premium": premium_q,
                    "Choice": choice,
                    "Quantity": regular_q + premium_q,
                }
            )
    return pd.DataFrame(rows)


OBS = build_observations()

# -----------------------------------------------------------------------------
# Independent household MNL
# -----------------------------------------------------------------------------
# Utilities are estimated separately for each household:
#   V_R = ASC_R + beta_R * z_R
#   V_P = ASC_P + beta_P * z_P
#   V_N = 0
# where z is the internally standardised price. No Purchase is the reference
# alternative, so its ASC is normalised to zero for identification.
#
# Only 11 observations are available for each household, and some alternatives
# are never observed. Unconstrained MLE can therefore diverge. Finite bounds
# keep the numerical optimisation stable and the UI explicitly flags estimates
# that hit those bounds.
# -----------------------------------------------------------------------------

PR_MEAN = OBS.P_Regular.mean()
PR_STD = OBS.P_Regular.std(ddof=0)
PP_MEAN = OBS.P_Premium.mean()
PP_STD = OBS.P_Premium.std(ddof=0)

PARAM_BOUNDS = [(-20.0, 20.0)] * 4
PARAM_NAMES = ["ASC_Regular", "ASC_Premium", "Beta_Regular_Price", "Beta_Premium_Price"]


def softmax(utilities):
    shifted = utilities - np.max(utilities, axis=1, keepdims=True)
    exp_u = np.exp(shifted)
    return exp_u / exp_u.sum(axis=1, keepdims=True)


def utility_matrix(params, p_regular, p_premium):
    asc_r, asc_p, beta_r, beta_p = params
    zr = (np.asarray(p_regular) - PR_MEAN) / PR_STD
    zp = (np.asarray(p_premium) - PP_MEAN) / PP_STD
    return np.column_stack([asc_r + beta_r * zr, asc_p + beta_p * zp, np.zeros_like(zr)])


def fit_household(household):
    sample = OBS[OBS.Household == household].copy().reset_index(drop=True)
    y = sample.Choice.map({"Regular": 0, "Premium": 1, "No Purchase": 2}).to_numpy()

    def negative_log_likelihood(params):
        probs = softmax(utility_matrix(params, sample.P_Regular, sample.P_Premium))
        chosen = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
        return float(-np.log(chosen).sum())

    starts = [
        np.array([0.0, 0.0, -1.0, -1.0]),
        np.array([1.0, 0.0, -1.0, -1.0]),
        np.array([0.0, 1.0, -1.0, -1.0]),
    ]
    results = [
        minimize(negative_log_likelihood, start, method="L-BFGS-B", bounds=PARAM_BOUNDS)
        for start in starts
    ]
    result = min(results, key=lambda r: r.fun)
    params = result.x
    probs = softmax(utility_matrix(params, sample.P_Regular, sample.P_Premium))
    predicted_idx = probs.argmax(axis=1)
    observed_idx = y

    at_bound = {
        PARAM_NAMES[i]: bool(
            abs(params[i] - PARAM_BOUNDS[i][0]) < 1e-5
            or abs(params[i] - PARAM_BOUNDS[i][1]) < 1e-5
        )
        for i in range(4)
    }

    return {
        "sample": sample,
        "result": result,
        "params": params,
        "probs": probs,
        "predicted": predicted_idx,
        "observed": observed_idx,
        "at_bound": at_bound,
        "nll": float(result.fun),
    }


MODELS = {household: fit_household(household) for household in HOUSEHOLDS}


def probabilities(model, p_regular, p_premium):
    probs = softmax(utility_matrix(model["params"], [p_regular], [p_premium]))[0]
    return {"Regular": float(probs[0]), "Premium": float(probs[1]), "No Purchase": float(probs[2])}


def quantity_summary(sample):
    regular = sample.loc[sample.Choice == "Regular", "Regular"]
    premium = sample.loc[sample.Choice == "Premium", "Premium"]
    return {
        "avg_all_weeks": float(sample.Quantity.mean()),
        "avg_regular_conditional": float(regular.mean()) if len(regular) else 0.0,
        "avg_premium_conditional": float(premium.mean()) if len(premium) else 0.0,
        "regular_n": int(len(regular)),
        "premium_n": int(len(premium)),
    }


def expected_quantity(model, sample, p_regular, p_premium):
    probs = probabilities(model, p_regular, p_premium)
    q = quantity_summary(sample)
    return probs["Regular"] * q["avg_regular_conditional"] + probs["Premium"] * q["avg_premium_conditional"]


def pairwise_boundaries(model):
    """Return 50% pairwise utility-equality boundaries in original price units."""
    asc_r, asc_p, beta_r, beta_p = model["params"]

    regular_np = PR_MEAN - asc_r * PR_STD / beta_r if abs(beta_r) > 1e-12 else np.nan
    premium_np = PP_MEAN - asc_p * PP_STD / beta_p if abs(beta_p) > 1e-12 else np.nan

    if abs(beta_p) > 1e-12:
        slope = (beta_r * PP_STD) / (beta_p * PR_STD)
        intercept = PP_MEAN + (asc_r - asc_p) * PP_STD / beta_p - slope * PR_MEAN
    else:
        slope, intercept = np.nan, np.nan

    return {
        "R_vs_N": (float(regular_np),),
        "P_vs_N": (float(premium_np),),
        "R_vs_P": (float(slope), float(intercept)),
    }


def boundary_text(model):
    b = pairwise_boundaries(model)
    r_np = b["R_vs_N"][0]
    p_np = b["P_vs_N"][0]
    slope, intercept = b["R_vs_P"]

    def fmt(x):
        return "not estimable" if not np.isfinite(x) else f"{x:.1f}"

    if np.isfinite(slope) and np.isfinite(intercept):
        sign = "+" if intercept >= 0 else "−"
        rp = f"P_P = {slope:.2f} × P_R {sign} {abs(intercept):.1f}"
    else:
        rp = "not estimable"
    return rp, f"P_R ≈ {fmt(r_np)}", f"P_P ≈ {fmt(p_np)}"


def decision_map(household, p_regular, p_premium):
    model = MODELS[household]
    sample = model["sample"]
    x_min, x_max = 30, 70
    y_min, y_max = 60, 110
    xs = np.linspace(x_min, x_max, 180)
    ys = np.linspace(y_min, y_max, 180)
    xx, yy = np.meshgrid(xs, ys)
    grid_probs = softmax(utility_matrix(model["params"], xx.ravel(), yy.ravel()))
    zones = grid_probs.argmax(axis=1).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=zones,
            zmin=0,
            zmax=2,
            colorscale=[
                [0.00, "rgba(37,99,235,0.10)"], [0.33, "rgba(37,99,235,0.10)"],
                [0.34, "rgba(216,155,0,0.10)"], [0.66, "rgba(216,155,0,0.10)"],
                [0.67, "rgba(100,116,139,0.10)"], [1.00, "rgba(100,116,139,0.10)"],
            ],
            showscale=False,
            hoverinfo="skip",
            name="Predicted choice region",
        )
    )

    b = pairwise_boundaries(model)
    r_np = b["R_vs_N"][0]
    p_np = b["P_vs_N"][0]
    slope, intercept = b["R_vs_P"]
    if np.isfinite(r_np) and x_min <= r_np <= x_max:
        fig.add_vline(x=r_np, line_width=2, line_dash="dash", line_color="#172033")
    if np.isfinite(p_np) and y_min <= p_np <= y_max:
        fig.add_hline(y=p_np, line_width=2, line_dash="dash", line_color="#172033")
    if np.isfinite(slope) and np.isfinite(intercept):
        line = slope * xs + intercept
        mask = np.isfinite(line) & (line >= y_min) & (line <= y_max)
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=xs[mask],
                    y=line[mask],
                    mode="lines",
                    name="Regular = Premium",
                    line=dict(color="#172033", width=3),
                    hovertemplate="<b>Regular = Premium</b><br>P<sub>P</sub>=%{y:.1f}<extra></extra>",
                )
            )

    for choice in CHOICES:
        sample_choice = sample[sample.Choice == choice]
        if sample_choice.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sample_choice.P_Regular,
                y=sample_choice.P_Premium,
                mode="markers",
                name=choice,
                marker=dict(size=12, color=COLORS[choice], line=dict(color="white", width=1.5)),
                customdata=sample_choice[["T", "Quantity"]].to_numpy(),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>Week %{customdata[0]}<br>"
                    "P<sub>R</sub>=%{x:.0f}<br>P<sub>P</sub>=%{y:.0f}<br>"
                    "Quantity=%{customdata[1]}<extra></extra>"
                ),
            )
        )

    scenario = probabilities(model, p_regular, p_premium)
    predicted = max(CHOICES, key=scenario.get)
    fig.add_trace(
        go.Scatter(
            x=[p_regular],
            y=[p_premium],
            mode="markers",
            name="Scenario",
            marker=dict(symbol="star", size=20, color="#172033", line=dict(color="white", width=2)),
            hovertemplate=(
                f"<b>Scenario</b><br>P<sub>R</sub>={p_regular:.0f}<br>P<sub>P</sub>={p_premium:.0f}<br>"
                f"Most likely: {predicted}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=590,
        margin=dict(l=10, r=10, t=18, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(title="Regular price (P_R)", range=[x_min, x_max], gridcolor="#E5E7EB", zeroline=False),
        yaxis=dict(title="Premium price (P_P)", range=[y_min, y_max], gridcolor="#E5E7EB", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    return fig, scenario


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container{max-width:1320px;padding-top:1.4rem;padding-bottom:3rem;}
section[data-testid="stSidebar"]{border-right:1px solid #E7E9EE;background:#FBFBFC;}
.hero{background:linear-gradient(135deg,#FFFDF7 0%,#F7F8FA 100%);border:1px solid #E7E9EE;border-radius:24px;padding:30px 34px;margin-bottom:20px;}
.kicker{font-size:.75rem;letter-spacing:.13em;text-transform:uppercase;font-weight:700;color:#667085;margin-bottom:8px;}
.hero h1{color:#172033;font-size:2.5rem;line-height:1.05;letter-spacing:-.045em;margin:0 0 10px;}
.hero p{color:#667085;font-size:1.02rem;margin:0;max-width:940px;}
.info-card{height:100%;border:1px solid #E7E9EE;border-radius:18px;background:#fff;padding:19px 20px;box-shadow:0 2px 8px rgba(23,32,51,.03);}
.info-card .label{font-size:.73rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:#667085;margin-bottom:7px;}
.info-card h4{margin:0 0 8px;color:#172033;font-size:1.02rem;}
.info-card p,.info-card li{color:#475467;font-size:.9rem;line-height:1.55;}
.callout{border-left:4px solid #172033;background:#F7F8FA;padding:14px 17px;border-radius:0 12px 12px 0;color:#344054;}
.small-note{font-size:.82rem;color:#667085;}
.metric-note{font-size:.8rem;color:#667085;margin-top:-10px;margin-bottom:12px;}
[data-testid="stMetric"]{border:1px solid #E7E9EE;border-radius:16px;padding:13px 15px;background:#fff;}
div[data-testid="stExpander"]{border:1px solid #E7E9EE;border-radius:14px;}
.eq{background:#F8FAFC;border:1px solid #E7E9EE;border-radius:14px;padding:15px 18px;margin:7px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#172033;}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    title_col, info_col = st.columns([0.84, 0.16], vertical_alignment="center")
    with title_col:
        st.markdown("### ☕ Choice Model Lab")
    with info_col:
        with st.popover("ⓘ"):
            st.markdown("### Model at a glance")
            st.caption("Independent multinomial logit (MNL) models estimated separately for each household.")
            st.markdown("**Decision-maker** — one household-week is one choice occasion.")
            st.markdown("**Alternatives** — Regular, Premium, and No Purchase.")
            st.markdown("**Inputs** — Regular price and Premium price. Household identity is not pooled into the estimation; each household has its own model.")
            st.markdown("**Outputs** — three probabilities that always sum to 100%, plus the most likely choice.")
            st.markdown("**Demand layer** — zero-purchase weeks remain zero in observed demand; scenario demand combines choice probabilities with observed quantity conditional on Regular/Premium purchase.")

    household = st.selectbox("Household", list(HOUSEHOLDS))
    st.divider()
    st.markdown("**Try a price scenario**")
    scenario_regular = st.slider("Regular price (P_R)", 30, 70, 42)
    scenario_premium = st.slider("Premium price (P_P)", 65, 105, 93)
    st.divider()
    st.caption("33 household-week observations · 11 weeks · 3 independent models")

model = MODELS[household]
sample = model["sample"]
scenario = probabilities(model, scenario_regular, scenario_premium)
prediction = max(CHOICES, key=scenario.get)
expected_q = expected_quantity(model, sample, scenario_regular, scenario_premium)
q_summary = quantity_summary(sample)

st.markdown(
    """
<div class="hero">
  <div class="kicker">Discrete choice demand experiment</div>
  <h1>How price changes household choice</h1>
  <p>A <b>separate multinomial logit model</b> for each household estimates the probability of choosing Regular, Premium, or No Purchase at any Regular/Premium price combination.</p>
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Most likely choice", prediction)
m2.metric("P(Regular)", f"{scenario['Regular']:.0%}")
m3.metric("P(Premium)", f"{scenario['Premium']:.0%}")
m4.metric("Expected capsules / week", f"{expected_q:.2f}")
st.markdown("<div class='metric-note'>Expected demand is zero-inclusive: No Purchase contributes 0 capsules.</div>", unsafe_allow_html=True)

st.subheader("Decision map")
st.caption("Dots are observed household-week choices. The background shows the alternative with the highest estimated utility. Pairwise lines mark equal-utility boundaries between alternatives.")
st.plotly_chart(decision_map(household, scenario_regular, scenario_premium)[0], use_container_width=True)

with st.expander("ⓘ How to read the map"):
    st.markdown(
        """
**1 · Each dot is an actual experiment observation.**  
Its coordinates are the Regular and Premium prices in that week. Its colour is the household's observed choice.

**2 · The background is a model prediction.**  
At every point on the price grid, the MNL calculates three probabilities and assigns the background to the alternative with the highest probability.

**3 · The boundaries are utility-equality lines.**  
They are not fitted trend lines. A boundary is the set of prices at which two alternatives have equal systematic utility.

**4 · Misclassification is possible.**  
An observed dot can sit in a different predicted region. With only 11 observations per household, that is expected and should be interpreted as model error/uncertainty rather than a bug.
"""
    )

st.subheader("Scenario simulator")
left, right = st.columns([1.15, 1])
with left:
    st.markdown(f"### {prediction}")
    st.write(f"At **P_R = {scenario_regular:.0f}** and **P_P = {scenario_premium:.0f}**, this household's estimated choice probabilities are:")
    st.progress(scenario["Regular"], text=f"Regular · {scenario['Regular']:.1%}")
    st.progress(scenario["Premium"], text=f"Premium · {scenario['Premium']:.1%}")
    st.progress(scenario["No Purchase"], text=f"No Purchase · {scenario['No Purchase']:.1%}")
with right:
    st.markdown("### Demand implication")
    st.metric("Observed zero-inclusive average", f"{q_summary['avg_all_weeks']:.2f} capsules/week")
    st.metric("Scenario expected demand", f"{expected_q:.2f} capsules/week")
    st.caption("Scenario demand uses the model's probability of buying each product multiplied by the observed average quantity conditional on buying that product. No Purchase contributes zero.")

# -----------------------------------------------------------------------------
# Model explanation
# -----------------------------------------------------------------------------
st.subheader("Estimated household model")
asc_r, asc_p, beta_r, beta_p = model["params"]
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='info-card'><div class='label'>Utility</div><h4>Random utility framework</h4><p>Each alternative has systematic utility plus an unobserved component. The household is modelled as choosing among alternatives according to their relative utility.</p></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='info-card'><div class='label'>Probability</div><h4>Multinomial logit</h4><p>The MNL converts the three utilities into probabilities using the softmax formula. The three probabilities therefore add to 100%.</p></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='info-card'><div class='label'>Estimation</div><h4>Maximum likelihood</h4><p>The parameters are selected to maximise the probability assigned to the choices that were actually observed in this household's 11-week experiment.</p></div>", unsafe_allow_html=True)

st.markdown("### Utility specification")
st.markdown("<div class='eq'>z_R = (P_R − 46.09) / 8.48 &nbsp;&nbsp;&nbsp; z_P = (P_P − 81.73) / 9.81</div>", unsafe_allow_html=True)
st.markdown(f"<div class='eq'>V<sub>R</sub> = {asc_r:.3f} + ({beta_r:.3f}) · z<sub>R</sub></div>", unsafe_allow_html=True)
st.markdown(f"<div class='eq'>V<sub>P</sub> = {asc_p:.3f} + ({beta_p:.3f}) · z<sub>P</sub></div>", unsafe_allow_html=True)
st.markdown("<div class='eq'>V<sub>N</sub> = 0 &nbsp;&nbsp; (reference alternative)</div>", unsafe_allow_html=True)
st.caption("The price means and standard deviations are calculated from the full 33-observation experiment and are only used to standardise prices numerically. The coefficients are estimated separately for the selected household.")

with st.expander("ⓘ Where do the numbers in these equations come from?"):
    st.markdown(
        f"""
### 1 · The price-standardisation numbers are calculated from the experiment

The model first converts prices into standardised variables so optimisation is numerically well behaved:

- **Regular:** z_R = (P_R − 46.09) / 8.48
- **Premium:** z_P = (P_P − 81.73) / 9.81

Those four numbers — **46.09, 8.48, 81.73, and 9.81** — are simply the mean and population standard deviation of the Regular and Premium prices across all 33 household-week observations. They are not estimated behavioural parameters.

### 2 · The household-specific coefficients are estimated from choices

For **{household}**, the optimiser estimates four parameters:

| Parameter | Estimated value | What it represents |
|---|---:|---|
| ASC_Regular | {asc_r:.3f} | Baseline utility of Regular relative to No Purchase |
| ASC_Premium | {asc_p:.3f} | Baseline utility of Premium relative to No Purchase |
| β_Regular | {beta_r:.3f} | How Regular utility changes when Regular price changes by 1 standard deviation |
| β_Premium | {beta_p:.3f} | How Premium utility changes when Premium price changes by 1 standard deviation |

The model chooses these values by **maximum likelihood**: it searches for the parameter combination that gives the highest probability to the actual 11 observed choices for this household.

### 3 · Why is No Purchase equal to zero?

This is an identification convention. We cannot estimate an absolute utility level for all three alternatives because adding the same constant to every utility leaves the MNL probabilities unchanged. So No Purchase is used as the reference and its systematic utility is fixed at 0.

That means each ASC is interpreted **relative to No Purchase**. For example, an ASC_Regular of 2 means that, at average Regular price, Regular has 2 more units of systematic utility than the No Purchase reference, before considering Premium.

### 4 · Where do the boundary numbers come from?

The boundaries are obtained by setting two utilities equal.

**Regular = No Purchase:**

V_R = 0 → ASC_R + β_R · z_R = 0

Solving for the original Regular price gives:

P_R = 46.09 − ASC_R × 8.48 / β_R

For {household}, this produces **P_R ≈ {boundary_text(model)[1].split('≈')[-1].strip()}**.

**Premium = No Purchase:**

V_P = 0 → ASC_P + β_P · z_P = 0

So:

P_P = 81.73 − ASC_P × 9.81 / β_P

For {household}, this produces **P_P ≈ {boundary_text(model)[2].split('≈')[-1].strip()}**.

**Regular = Premium:**

V_R = V_P

After substituting the two standardisation equations and rearranging, the result is a straight line:

P_P = slope × P_R + intercept

where the slope and intercept are functions of the two ASCs, the two price coefficients, and the price means/standard deviations.

So the **0.67, 39.7, 60.8, and 80.4** you may see for Household 1 are not manually entered numbers. They are calculated directly from Household 1's fitted coefficients and the experiment's price scaling constants.

### 5 · One important caveat

These coefficients are **exploratory estimates**, not precise economic measurements. Each household has only 11 observations, and some alternatives are never observed for particular households. When an estimate reaches the ±20 numerical bound, that indicates separation or weak identification. The app flags this below rather than treating the coefficient as a reliable behavioural estimate.
"""
    )

st.markdown("### All household estimates")
rows = []
for hh, hh_model in MODELS.items():
    ar, ap, br, bp = hh_model["params"]
    rows.append(
        {
            "Household": hh,
            "ASC Regular": ar,
            "ASC Premium": ap,
            "β Regular price": br,
            "β Premium price": bp,
            "Bounded parameters": ", ".join(k for k, v in hh_model["at_bound"].items() if v) or "None",
        }
    )
param_table = pd.DataFrame(rows)
for col in ["ASC Regular", "ASC Premium", "β Regular price", "β Premium price"]:
    param_table[col] = param_table[col].map(lambda x: f"{x:.3f}")
st.dataframe(param_table, use_container_width=True, hide_index=True)
st.caption("These are the actual household-specific estimates used by the simulator. A bounded parameter is a numerical warning, not evidence of a precise economic effect.")

st.markdown("### Choice probabilities")
st.markdown("<div class='eq'>P(i) = exp(V<sub>i</sub>) / [exp(V<sub>R</sub>) + exp(V<sub>P</sub>) + exp(V<sub>N</sub>)]</div>", unsafe_allow_html=True)
st.caption("The household chooses among Regular, Premium, and No Purchase. Changing either price changes relative utility and therefore all three probabilities.")

st.markdown("### Pairwise boundaries")
rp, rn, pn = boundary_text(model)
b1, b2, b3 = st.columns(3)
with b1:
    st.markdown(f"**Regular = Premium**  \n`{rp}`")
with b2:
    st.markdown(f"**Regular = No Purchase**  \n`{rn}`")
with b3:
    st.markdown(f"**Premium = No Purchase**  \n`{pn}`")

with st.expander("ⓘ What do these equations mean?"):
    st.markdown(
        """
The **Regular = Premium** boundary is where the two alternatives have the same systematic utility. Because the utility specification uses price linearly, it can be written as a straight line in the P_R × P_P map.

The **Regular = No Purchase** and **Premium = No Purchase** boundaries are the prices where the corresponding product has the same utility as the outside option. In this specification they appear as a vertical or horizontal threshold because No Purchase has a normalised utility of zero.

These are **choice-model boundaries**, not statements that a household has a single causal willingness-to-pay threshold. The current experiment is small and the independent household estimates are deliberately treated as exploratory.
"""
    )

st.subheader("Observed data")
obs_display = sample[["T", "P_Regular", "P_Premium", "Choice", "Quantity"]].copy()
obs_display.columns = ["Week", "P_R", "P_P", "Observed choice", "Quantity"]
st.dataframe(obs_display, use_container_width=True, hide_index=True)

st.subheader("Model diagnostics")
accuracy = float((model["predicted"] == model["observed"]).mean())
choice_counts = sample.Choice.value_counts().reindex(CHOICES, fill_value=0)
d1, d2, d3, d4 = st.columns(4)
d1.metric("Observed choices", f"{len(sample)}")
d2.metric("In-sample accuracy", f"{accuracy:.0%}")
d3.metric("Log-likelihood", f"{-model['nll']:.3f}")
d4.metric("Parameters", "4")

# BUG FIX: dict.values is a method; call it with parentheses.
if any(model["at_bound"].values()):
    bounded = [name for name, hit in model["at_bound"].items() if hit]
    st.warning(
        "At least one parameter reached the finite estimation bound "
        f"({', '.join(bounded)}). This is a diagnostic of separation/weak identification, "
        "not evidence of an economically precise coefficient. The app keeps the estimate "
        "finite so the household-level simulator can remain explorable."
    )

with st.expander("ⓘ Why is this model intentionally independent?"):
    st.markdown(
        """
This version does **not pool households**. Household 1, Household 2, and Household 3 each receive their own MNL estimated only from that household's 11 weekly observations.

That choice matches the experimental question: **how does this particular household respond to the price pair?** It also makes the simulator directly interpretable at the household level.

The trade-off is important: each model has only 11 observations. In addition, Household 1 and Household 2 never observed No Purchase, while Household 3 never observed Regular. This creates separation or weak identification in a conventional unconstrained MLE. The app therefore uses bounded numerical optimisation and flags estimates that reach a bound.
"""
    )

with st.expander("ⓘ Why MNL instead of the previous classifier?"):
    st.markdown(
        """
The previous app used two binary logistic classifiers in sequence: Buy vs No Purchase, then Regular vs Premium conditional on buying.

The MNL instead treats **Regular, Premium, and No Purchase as three competing alternatives in one Random Utility Model**. The model asks what relative utility structure could have generated the observed choice, rather than fitting two separate classification problems.

This gives us a natural economic interpretation of utility, alternative-specific constants, price sensitivity, joint choice probabilities, and pairwise utility boundaries.
"""
    )

with st.expander("ⓘ What is included in demand and what is not?"):
    st.markdown(
        """
**Included:**
- Zero-purchase weeks as genuine observations of zero demand.
- Zero-inclusive observed average weekly demand.
- Regular and Premium choice probabilities from the MNL.
- Expected scenario demand using observed conditional quantities.
- Household-specific estimation rather than pooled household effects.

**Not included:**
- A structural causal elasticity model.
- A validated willingness-to-pay estimate.
- A separate behavioural model of consumption intensity beyond the simple conditional quantity layer.
- Out-of-sample validation; with 11 observations per household, the displayed accuracy is in-sample.
"""
    )

st.markdown("### Model reference")
st.caption("The MNL formulation follows the standard discrete-choice framework: alternative-specific constants are identified relative to a reference alternative, utilities are converted to choice probabilities through the logit formula, and parameters are estimated by maximum likelihood. See the project README for equations, assumptions, interpretation, and limitations.")

st.divider()
st.caption("Coffee Capsule Choice Model · Independent household MNL · Exploratory research tool")
