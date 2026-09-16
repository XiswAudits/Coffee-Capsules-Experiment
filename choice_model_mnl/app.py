import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="Coffee Capsules — MNL Choice Model", page_icon="☕", layout="wide", initial_sidebar_state="collapsed")
RAW=pd.read_csv("coffee_capsules_data.csv")
HOUSEHOLDS={"Household 1":("HH1_Regular","HH1_Premium"),"Household 2":("HH2_Regular","HH2_Premium"),"Household 3":("HH3_Regular","HH3_Premium")}
CHOICES=["Regular","Premium","No Purchase"]
COLORS={"Regular":"#1677ff","Premium":"#ff8a1f","No Purchase":"#aab4c0"}
BASELINE_R,BASELINE_P=42,93
PARAMETER_BOUNDS=[(-20.0,20.0)]*4
PARAMETER_NAMES=["ASC_Regular","ASC_Premium","Beta_Regular_Price","Beta_Premium_Price"]

def build_observations():
    rows=[]
    for h,(rc,pc) in HOUSEHOLDS.items():
        for _,r in RAW.iterrows():
            rq,pq=int(r[rc]),int(r[pc]);c="Regular" if rq>0 else "Premium" if pq>0 else "No Purchase"
            rows.append({"Household":h,"T":int(r["T"]),"P_Regular":float(r["P_Regular"]),"P_Premium":float(r["P_Premium"]),"Regular":rq,"Premium":pq,"Choice":c,"Quantity":rq+pq})
    return pd.DataFrame(rows)
OBS=build_observations();PR_MEAN,PR_STD=OBS.P_Regular.mean(),OBS.P_Regular.std(ddof=0);PP_MEAN,PP_STD=OBS.P_Premium.mean(),OBS.P_Premium.std(ddof=0)

def softmax(utilities):
    shifted=utilities-np.max(utilities,axis=1,keepdims=True);exp_u=np.exp(shifted);return exp_u/exp_u.sum(axis=1,keepdims=True)

def utility(parameters,regular_price,premium_price):
    ar,ap,br,bp=parameters;zr=(np.asarray(regular_price)-PR_MEAN)/PR_STD;zp=(np.asarray(premium_price)-PP_MEAN)/PP_STD
    return np.column_stack([ar+br*zr,ap+bp*zp,np.zeros_like(zr)])

def fit_household(household):
    sample=OBS[OBS.Household==household].reset_index(drop=True);observed=sample.Choice.map({"Regular":0,"Premium":1,"No Purchase":2}).to_numpy()
    def nll(p):
        probs=softmax(utility(p,sample.P_Regular,sample.P_Premium));chosen=probs[np.arange(len(observed)),observed];return float(-np.log(np.clip(chosen,1e-15,1)).sum())
    starts=[np.array([0.,0.,-1.,-1.]),np.array([1.,0.,-1.,-1.]),np.array([0.,1.,-1.,-1.])]
    results=[minimize(nll,start,method="L-BFGS-B",bounds=PARAMETER_BOUNDS) for start in starts];result=min(results,key=lambda x:x.fun);p=result.x;probs=softmax(utility(p,sample.P_Regular,sample.P_Premium))
    at_bound={PARAMETER_NAMES[i]:bool(abs(p[i]-PARAMETER_BOUNDS[i][0])<1e-5 or abs(p[i]-PARAMETER_BOUNDS[i][1])<1e-5) for i in range(4)}
    return {"sample":sample,"params":p,"probs":probs,"predicted":probs.argmax(axis=1),"observed":observed,"at_bound":at_bound,"nll":float(result.fun),"success":bool(result.success),"observed_counts":sample.Choice.value_counts().to_dict()}
MODELS={h:fit_household(h) for h in HOUSEHOLDS}

def probabilities(model,regular_price,premium_price):
    q=softmax(utility(model["params"],[regular_price],[premium_price]))[0];return dict(zip(CHOICES,q.astype(float)))

def boundaries(model):
    ar,ap,br,bp=model["params"];rn=PR_MEAN-ar*PR_STD/br if abs(br)>1e-12 else np.nan;pn=PP_MEAN-ap*PP_STD/bp if abs(bp)>1e-12 else np.nan
    if abs(bp)>1e-12:slope=br*PP_STD/(bp*PR_STD);intercept=PP_MEAN+(ar-ap)*PP_STD/bp-slope*PR_MEAN
    else:slope,intercept=np.nan,np.nan
    return rn,pn,slope,intercept

def quantity_stats(frame):
    regular=frame.loc[frame.Choice=="Regular","Regular"];premium=frame.loc[frame.Choice=="Premium","Premium"]
    return float(frame.Quantity.mean()),float(regular.mean()) if len(regular) else 0.,float(premium.mean()) if len(premium) else 0.

def expected_revenue_household(model,regular_price,premium_price):
    scenario=probabilities(model,regular_price,premium_price)
    _,regular_qty,premium_qty=quantity_stats(model["sample"])
    regular_revenue=scenario["Regular"]*regular_price*regular_qty
    premium_revenue=scenario["Premium"]*premium_price*premium_qty
    return {"Regular":regular_revenue,"Premium":premium_revenue,"Total":regular_revenue+premium_revenue,"Regular_Qty":regular_qty,"Premium_Qty":premium_qty}

def expected_revenue_all(regular_price,premium_price):
    household_revenue={h:expected_revenue_household(model,regular_price,premium_price) for h,model in MODELS.items()}
    total=sum(r["Total"] for r in household_revenue.values())
    return household_revenue,total

def decision_map(household,regular_price,premium_price):
    model=MODELS[household];sample=model["sample"];xs=np.linspace(30,70,180);ys=np.linspace(55,145,180);xx,yy=np.meshgrid(xs,ys);probs=softmax(utility(model["params"],xx.ravel(),yy.ravel()));region=probs.argmax(axis=1).reshape(xx.shape)
    fig=go.Figure(go.Heatmap(x=xs,y=ys,z=region,zmin=0,zmax=2,colorscale=[[0,"rgba(22,119,255,.12)"],[.33,"rgba(22,119,255,.12)"],[.34,"rgba(255,138,31,.10)"],[.66,"rgba(255,138,31,.10)"],[.67,"rgba(170,180,192,.12)"],[1,"rgba(170,180,192,.12)"]],showscale=False,hoverinfo="skip"))
    rn,pn,slope,intercept=boundaries(model)
    if np.isfinite(rn) and 30<=rn<=70:fig.add_vline(x=rn,line_width=2,line_dash="dash",line_color="#1677ff")
    if np.isfinite(pn) and 55<=pn<=145:fig.add_hline(y=pn,line_width=2,line_dash="dash",line_color="#ff8a1f")
    if np.isfinite(slope) and np.isfinite(intercept):
        line=slope*xs+intercept;mask=(line>=55)&(line<=145)
        if mask.any():fig.add_trace(go.Scatter(x=xs[mask],y=line[mask],mode="lines",name="Regular = Premium",line={"color":"#172033","width":2}))
    for choice in CHOICES:
        data=sample[sample.Choice==choice]
        if data.empty:continue
        fig.add_trace(go.Scatter(x=data.P_Regular,y=data.P_Premium,mode="markers",name=choice,marker={"size":10,"color":COLORS[choice],"line":{"color":"white","width":1.5}},customdata=data[["T","Quantity"]].to_numpy(),hovertemplate="<b>%{fullData.name}</b><br>Week %{customdata[0]}<br>Regular €%{x:.0f}<br>Premium €%{y:.0f}<br>Quantity %{customdata[1]}<extra></extra>"))
    scenario=probabilities(model,regular_price,premium_price);predicted=max(CHOICES,key=scenario.get);fig.add_trace(go.Scatter(x=[regular_price],y=[premium_price],mode="markers",name="Scenario",marker={"symbol":"star","size":18,"color":"#172033","line":{"color":"white","width":2}}))
    fig.update_layout(height=500,margin={"l":10,"r":10,"t":20,"b":10},paper_bgcolor="white",plot_bgcolor="white",font={"family":"Inter, Arial, sans-serif","color":"#172033"},xaxis={"title":"Regular price (€)","range":[30,70],"gridcolor":"#e8edf3","zeroline":False,"fixedrange":True},yaxis={"title":"Premium price (€)","range":[55,145],"gridcolor":"#e8edf3","zeroline":False,"fixedrange":True},legend={"orientation":"h","y":1.04,"x":0},hoverlabel={"bgcolor":"white"})
    return fig,scenario,predicted

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--ink:#101828;--muted:#667085;--line:#e5eaf0;--page:#f3f5f7;--blue:#1677ff;--orange:#ff8a1f}
html,body,[class*="css"],.stApp{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important}.stApp,[data-testid="stAppViewContainer"]{background:var(--page);color:var(--ink)}[data-testid="stSidebar"]{display:none}.block-container{max-width:1460px;padding:20px 28px 54px}
.top-shell{background:rgba(255,255,255,.98);border:1px solid var(--line);border-radius:22px;padding:13px 18px;margin-bottom:16px;box-shadow:0 8px 24px rgba(16,24,40,.035)}.brand{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:700;color:#172033}.brand-icon{width:28px;height:28px;border-radius:8px;display:grid;place-items:center;color:white;background:linear-gradient(145deg,#ffb21a,#ff8a1f);font-size:16px}.nav{display:flex;gap:26px;align-items:center;justify-content:center}.nav a{color:#344054;text-decoration:none;font-size:13px}.nav a.active{background:#172033;color:white;padding:9px 16px;border-radius:11px;box-shadow:0 4px 12px rgba(16,24,40,.12)}.profile{width:34px;height:34px;border:1px solid var(--line);border-radius:50%;display:grid;place-items:center;color:#344054}
.scenario-shell{background:#fff;border:1px solid var(--line);border-radius:18px;padding:13px 16px;margin-bottom:10px;box-shadow:0 5px 18px rgba(16,24,40,.025)}.scenario-title{font-size:12px;font-weight:600;color:#344054;margin-bottom:4px}.scenario-help{font-size:11px;color:#98a2b3}.scenario-shell [data-testid="stSlider"]{padding-bottom:0}.scenario-shell [data-testid="stWidgetLabel"]{font-size:11px!important;color:#667085!important}.hero{margin:20px 0 24px}.hero-copy{max-width:780px}.hero h1{font-size:38px;line-height:1.05;letter-spacing:-1.8px;margin:0 0 12px;color:#101828;font-weight:600}.hero p{margin:0;color:#667085;font-size:14px;line-height:1.6}.card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 5px 20px rgba(16,24,40,.025);height:100%}.card-title{color:#101828;font-size:13px;font-weight:600;margin-bottom:5px}.card-sub{color:#98a2b3;font-size:11px;line-height:1.5}.kpi-value{color:#101828;font-size:26px;font-weight:600;letter-spacing:-1px;margin-top:6px}.insight{background:linear-gradient(145deg,#172033,#223a53);color:#fff;border-radius:18px;padding:20px;min-height:180px;box-shadow:0 14px 30px rgba(23,32,51,.14)}.insight .eyebrow{background:rgba(255,255,255,.10);color:#dbeafe}.insight h2{color:white;font-size:27px;margin:5px 0;letter-spacing:-1px}.insight p{color:#cbd5e1;font-size:12px;margin:0}.prob-row{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.10);font-size:12px}.prob-row:last-child{border-bottom:0}.pill{display:inline-block;padding:5px 9px;border-radius:999px;font-size:10px;font-weight:600;background:#f2f4f7;color:#475467;margin:2px 2px 0 0}.equation{background:#f8fafc;border:1px solid #e7edf3;border-radius:12px;padding:11px 13px;margin-top:9px;color:#172033;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}.warning{background:#fff8ed;border:1px solid #f6d9ad;border-radius:14px;padding:13px 15px;color:#7a4a0b;font-size:11px;line-height:1.5}[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:14px 15px;box-shadow:0 4px 16px rgba(16,24,40,.025)}[data-testid="stMetricLabel"]{font-size:11px!important;color:#667085!important}[data-testid="stMetricValue"]{font-size:25px!important;color:#101828!important}div[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:#fff}[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden}
@media(max-width:900px){.block-container{padding:16px}.nav{gap:13px}.hero h1{font-size:32px}.scenario-shell{padding:12px}}
@media(max-width:640px){.block-container{padding:10px}.top-shell{border-radius:18px;padding:12px}.nav{display:none}.brand{font-size:16px}.hero h1{font-size:27px}.hero p{font-size:13px}.card{border-radius:15px;padding:14px}.insight{border-radius:15px}[data-testid="stMetric"]{padding:11px 12px}[data-testid="stMetricValue"]{font-size:21px!important}.js-plotly-plot .plotly{width:100%!important}}
</style>
""",unsafe_allow_html=True)

st.markdown('<div class="top-shell"><div style="display:flex;align-items:center;justify-content:space-between;gap:18px"><div class="brand"><span class="brand-icon">☕</span> Coffee Capsule Lab</div><div class="nav"><a class="active" href="#home">Home</a><a href="#model">Model</a><a href="#data">Data</a><a href="#methodology">Methodology</a><a href="#insights">Insights</a></div><div class="profile">◦</div></div></div><div id="home"></div>',unsafe_allow_html=True)
st.markdown('<div class="scenario-shell"><div class="scenario-title">Scenario controls</div><div class="scenario-help">Choose a household and test a Regular / Premium price pair. These controls are intentionally separated from the navigation.</div></div>',unsafe_allow_html=True)
sc1,sc2,sc3=st.columns([1.1,1,1])
with sc1: household=st.selectbox("Household",list(HOUSEHOLDS),key="mnl_household")
with sc2: regular_price=st.slider("Regular price (€)",30,70,BASELINE_R,key="mnl_regular_price")
with sc3: premium_price=st.slider("Premium price (€)",65,105,BASELINE_P,key="mnl_premium_price")

fig,scenario,predicted=decision_map(household,regular_price,premium_price);model=MODELS[household];avg_qty,regular_qty,premium_qty=quantity_stats(model["sample"]);revenue=expected_revenue_household(model,regular_price,premium_price);household_revenues,total_expected_revenue=expected_revenue_all(regular_price,premium_price);rn,pn,slope,intercept=boundaries(model);asc_r,asc_p,beta_r,beta_p=model["params"];hits=[n for n,v in model["at_bound"].items() if v];counts=model["observed_counts"]
st.markdown(f'<div class="hero"><div class="hero-copy"><div class="eyebrow">Multinomial logit model</div><h1>Household Choice<br>Model</h1><p>Estimate choice probabilities, expected revenue, and equal-utility boundaries for Regular, Premium, and No Purchase using an independent MNL for each household.</p></div></div>',unsafe_allow_html=True)
m1,m2,m3,m4=st.columns(4);m1.metric("Most likely choice",predicted);m2.metric("P(Regular)",f"{scenario['Regular']:.0%}");m3.metric("P(Premium)",f"{scenario['Premium']:.0%}");m4.metric("P(No Purchase)",f"{scenario['No Purchase']:.0%}")
st.markdown('<div id="model"></div>',unsafe_allow_html=True);left,right=st.columns([1.65,.8],gap="medium")
with left: st.markdown('<div class="card"><div class="card-title">Choice map</div><div class="card-sub">Observed choices, model-implied utility boundaries, and the current price scenario.</div></div>',unsafe_allow_html=True);st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False,"responsive":True})
with right: st.markdown(f'<div class="insight"><div class="eyebrow">Scenario result</div><h2>{predicted}</h2><p>At Regular €{regular_price} and Premium €{premium_price}</p><div style="height:12px"></div><div class="prob-row"><span>Regular</span><b>{scenario["Regular"]:.1%}</b></div><div class="prob-row"><span>Premium</span><b>{scenario["Premium"]:.1%}</b></div><div class="prob-row"><span>No Purchase</span><b>{scenario["No Purchase"]:.1%}</b></div></div>',unsafe_allow_html=True);st.write("");st.markdown(f'<div class="card"><div class="card-title">Expected revenue</div><div class="card-sub">{household} expected revenue at the selected price scenario.</div><div class="kpi-value">€{revenue["Total"]:.2f}</div><div class="card-sub">selected household</div><div style="height:8px"></div><span class="pill">Regular €{revenue["Regular"]:.2f}</span><span class="pill">Premium €{revenue["Premium"]:.2f}</span><div class="card-sub" style="margin-top:9px;">Across all 3 households: <b>€{total_expected_revenue:.2f}</b></div></div>',unsafe_allow_html=True)

st.markdown('<div id="insights"></div>',unsafe_allow_html=True);i1,i2,i3=st.columns(3)
with i1: st.markdown('<div class="card"><div class="card-title">Utility specification</div><div class="card-sub">No Purchase is the reference alternative.</div><div class="equation">Vᵣ = ASCᵣ + βᵣ · zᵣ</div><div class="equation">Vₚ = ASCₚ + βₚ · zₚ</div><div class="equation">Vₙ = 0</div></div>',unsafe_allow_html=True)
with i2: st.markdown(f'<div class="card"><div class="card-title">Selected household parameters</div><div class="card-sub">{household}</div><div class="kpi-value">{asc_r:.2f} / {asc_p:.2f}</div><div class="card-sub">ASC Regular / Premium</div><div class="kpi-value">{beta_r:.2f} / {beta_p:.2f}</div><div class="card-sub">Price coefficient Regular / Premium</div></div>',unsafe_allow_html=True)
with i3: st.markdown(f'<div class="card"><div class="card-title">Choice boundaries</div><div class="card-sub">Equal-utility thresholds in original price units.</div><div class="equation">Regular = Premium<br>Pₚ = {slope:.2f} × Pᵣ + {intercept:.1f}</div><div class="card-sub" style="margin-top:9px;">Regular = No Purchase: {rn:.1f} €</div><div class="card-sub">Premium = No Purchase: {pn:.1f} €</div></div>',unsafe_allow_html=True)

st.markdown('<div class="card" style="margin-top:16px;"><div class="card-title">Expected revenue equation</div><div class="card-sub">First calculate expected revenue for each household at the selected prices, then sum across households.</div><div class="equation">E[Revenueₕ] = Pₕ(Regular) × Pᵣ × E[Qₕ | Regular] + Pₕ(Premium) × Pₚ × E[Qₕ | Premium]</div><div class="equation">E[Revenueₜₒₜₐₗ] = Σₕ E[Revenueₕ]</div><div class="card-sub" style="margin-top:9px;">No Purchase contributes €0. The quantity terms are observed conditional average quantities for each household, so this is an illustrative revenue layer rather than a structural quantity-demand model.</div></div>',unsafe_allow_html=True)

st.markdown('<div class="card" style="margin-top:16px;"><div class="card-title">Household revenue breakdown</div><div class="card-sub">Expected revenue at the current price scenario for each household.</div></div>',unsafe_allow_html=True)
rev_rows=[]
for h,r in household_revenues.items():
    rev_rows.append({"Household":h,"Expected Regular Revenue":round(r["Regular"],2),"Expected Premium Revenue":round(r["Premium"],2),"Expected Total Revenue":round(r["Total"],2)})
rev_rows.append({"Household":"Total","Expected Regular Revenue":round(sum(r["Regular"] for r in household_revenues.values()),2),"Expected Premium Revenue":round(sum(r["Premium"] for r in household_revenues.values()),2),"Expected Total Revenue":round(total_expected_revenue,2)})
st.dataframe(pd.DataFrame(rev_rows),use_container_width=True,hide_index=True)

if hits: st.markdown(f'<div class="warning"><b>Identification warning.</b> {", ".join(hits)} reached the ±20 optimisation bound. With only 11 observations and missing choice categories, these parameters are weakly identified. Treat the associated estimates as exploratory rather than precise economic effects.</div>',unsafe_allow_html=True)

st.markdown('<div id="methodology"></div>',unsafe_allow_html=True);st.subheader("Model & methodology")
with st.expander("Step-by-step explanation"):
    st.markdown(f"""### 1. Start with the original data
- 11 weeks of observations for each of 3 households.
- Each household therefore contributes **11 observations** to its own MNL.
- Across the dataset there are **33 household-week observations**.

### 2. Convert quantities into observed choices
For every household-week:
- Regular quantity > 0 → **Regular**
- Otherwise Premium quantity > 0 → **Premium**
- Otherwise → **No Purchase**

The MNL therefore models three alternatives: **Regular, Premium, No Purchase**.

### 3. Estimate one MNL separately for each household
The dashboard fits an independent model for each household, giving each household its own ASC_Regular, ASC_Premium, Regular-price coefficient, and Premium-price coefficient.

### 4. Standardize the prices
`Z_R = (P_R − mean(P_R)) / SD(P_R)`  
`Z_P = (P_P − mean(P_P)) / SD(P_P)`

Across the 33 observations: Regular mean ≈ **€46.09**, SD ≈ **€8.48**; Premium mean ≈ **€81.73**, SD ≈ **€9.81**.

### 5. Define utility for each alternative
No Purchase is the reference alternative:

`V_R = ASC_R + β_R × Z_R`  
`V_P = ASC_P + β_P × Z_P`  
`V_N = 0`

The ASCs capture baseline preference relative to No Purchase; the price coefficients determine how standardized prices enter utility.

### 6. Convert utilities into probabilities using softmax
`P(i) = exp(V_i) / [exp(V_R) + exp(V_P) + exp(V_N)]`

### 7. Compare predicted probabilities with observed choices
For each week, the model takes the probability assigned to the choice that actually occurred.

### 8. Build the negative log-likelihood
`NLL = −Σ log(P_observed)`

Lower NLL means the model assigns greater probability to the observed choices.

### 9. Estimate the parameters numerically
There is no closed-form MNL solution, so the model uses **L-BFGS-B** numerical optimization, parameter bounds of **−20 to +20**, and three different starting points. The result with the lowest NLL is retained.

### 10. Calculate scenario probabilities
For the selected household and prices, the fitted utilities are converted into probabilities for Regular, Premium, and No Purchase.

### 11. Calculate expected revenue per household
For each household at the selected price pair:

`E[Revenue_h] = P_h(Regular) × P_Regular × E[Q_h | Regular] + P_h(Premium) × P_Premium × E[Q_h | Premium]`

No Purchase contributes zero revenue.

### 12. Aggregate expected revenue across households
`E[Revenue_Total] = Σ_h E[Revenue_h]`

The dashboard therefore calculates household-level expected revenue first and then sums the three household values to obtain total expected revenue.

### 13. Derive the equal-utility boundaries
**Regular = No Purchase:** set `V_R = V_N = 0`.  
**Premium = No Purchase:** set `V_P = V_N = 0`.  
**Regular = Premium:** set `V_R = V_P`.

The resulting equations are transformed back from standardized prices into euros for the chart.

### 14. Limitations
The independent household MNLs are based on only 11 observations per household. Some alternatives are never observed for some households, creating weak identification. The MNL also relies on the IIA assumption. The model is exploratory and the quantity/revenue layer is not a structural demand model.""")
with st.expander("How should I read the boundaries?"): st.markdown("""- **Dashed vertical line:** Regular = No Purchase.
- **Dashed horizontal line:** Premium = No Purchase.
- **Solid diagonal line:** Regular = Premium.
- **Dots:** observed household-week choices.
- **Star:** current scenario.

These are **equal-utility model boundaries**, not validated willingness-to-pay thresholds.""")
with st.expander("Why independent household MNLs?"): st.markdown("""The independent specification lets the dashboard estimate a distinct preference structure for each household. The trade-off is statistical power: each household has only 11 observations, and some alternatives are never observed.

This means the model should be treated as an exploratory discrete-choice exercise rather than a validated causal demand model. Standard MNL assumptions, including IIA, should also be kept in mind.""")
st.markdown('<div id="data"></div>',unsafe_allow_html=True);st.subheader("Observed weeks");sample=model["sample"][["T","P_Regular","P_Premium","Choice","Quantity"]].copy();sample.columns=["Week","Regular price","Premium price","Observed choice","Quantity"];st.dataframe(sample,use_container_width=True,hide_index=True);st.caption("33 total household-week observations · 11 observations per household · independent MNL estimates")