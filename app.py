import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Coffee Capsules — Demand Model", page_icon="☕", layout="wide", initial_sidebar_state="collapsed")
RAW = pd.read_csv("coffee_capsules_data.csv")
HOUSEHOLDS = {"Household 1": ("HH1_Regular", "HH1_Premium"), "Household 2": ("HH2_Regular", "HH2_Premium"), "Household 3": ("HH3_Regular", "HH3_Premium")}
CHOICES = ["Regular", "Premium", "No Purchase"]
COLORS = {"Regular":"#1677ff", "Premium":"#ff8a1f", "No Purchase":"#aab4c0"}
BASELINE_R, BASELINE_P = 42, 93

def build_observations():
    rows=[]
    for h,(rc,pc) in HOUSEHOLDS.items():
        for _,r in RAW.iterrows():
            rq,pq=int(r[rc]),int(r[pc]); c="Regular" if rq>0 else "Premium" if pq>0 else "No Purchase"
            rows.append({"Household":h,"T":int(r["T"]),"P_Regular":float(r["P_Regular"]),"P_Premium":float(r["P_Premium"]),"Regular":rq,"Premium":pq,"Choice":c,"Quantity":rq+pq})
    return pd.DataFrame(rows)
OBS=build_observations(); PR_MEAN,PR_STD=OBS.P_Regular.mean(),OBS.P_Regular.std(ddof=0); PP_MEAN,PP_STD=OBS.P_Premium.mean(),OBS.P_Premium.std(ddof=0)

def features(f):
    return np.column_stack([(f.P_Regular.to_numpy()-PR_MEAN)/PR_STD,(f.P_Premium.to_numpy()-PP_MEAN)/PP_STD,(f.Household=="Household 2").astype(float).to_numpy(),(f.Household=="Household 3").astype(float).to_numpy()])
purchase_model=LogisticRegression(C=1,max_iter=5000).fit(features(OBS),(OBS.Choice!="No Purchase").astype(int)); BUYERS=OBS[OBS.Choice!="No Purchase"]; choice_model=LogisticRegression(C=1,max_iter=5000).fit(features(BUYERS),(BUYERS.Choice=="Premium").astype(int))

def probabilities(h,pr,pp):
    x=pd.DataFrame({"Household":[h],"P_Regular":[pr],"P_Premium":[pp]}); buy=purchase_model.predict_proba(features(x))[0,1]; prem=choice_model.predict_proba(features(x))[0,1]
    return {"No Purchase":1-buy,"Regular":buy*(1-prem),"Premium":buy*prem,"Buy":buy,"Premium | Buy":prem}

def boundary(model,h):
    c=model.coef_[0]; he=c[2] if h=="Household 2" else c[3] if h=="Household 3" else 0
    if abs(c[1])<1e-12:return np.nan,np.nan
    sz=-c[0]/c[1]; iz=-(model.intercept_[0]+he)/c[1]; slope=sz*PP_STD/PR_STD; intercept=PP_MEAN+PP_STD*iz-slope*PR_MEAN
    return float(slope),float(intercept)

def decision_map(h,pr,pp):
    ps,pi=boundary(purchase_model,h); cs,ci=boundary(choice_model,h); xs=np.linspace(30,70,180); ys=np.linspace(55,145,180); xx,yy=np.meshgrid(xs,ys); grid=pd.DataFrame({"Household":h,"P_Regular":xx.ravel(),"P_Premium":yy.ravel()}); buy=purchase_model.predict_proba(features(grid))[:,1].reshape(xx.shape); prem=choice_model.predict_proba(features(grid))[:,1].reshape(xx.shape); z=np.where(buy<.5,2,np.where(prem>=.5,1,0))
    fig=go.Figure(go.Heatmap(x=xs,y=ys,z=z,zmin=0,zmax=2,colorscale=[[0,"rgba(22,119,255,.12)"],[.33,"rgba(22,119,255,.12)"],[.34,"rgba(255,138,31,.10)"],[.66,"rgba(255,138,31,.10)"],[.67,"rgba(170,180,192,.12)"],[1,"rgba(170,180,192,.12)"]],showscale=False,hoverinfo="skip"))
    for slope,intercept,name,dash,color in [(ps,pi,"Purchase boundary","dash","#1677ff"),(cs,ci,"Regular / Premium","dash","#ff8a1f")]:
        if np.isfinite(slope) and np.isfinite(intercept):
            line=slope*xs+intercept; mask=np.isfinite(line)&(line>=ys.min())&(line<=ys.max())
            if mask.any(): fig.add_trace(go.Scatter(x=xs[mask],y=line[mask],mode="lines",name=name,line={"color":color,"width":2,"dash":dash}))
    hd=OBS[OBS.Household==h]
    for ch in CHOICES:
        s=hd[hd.Choice==ch]
        if not s.empty: fig.add_trace(go.Scatter(x=s.P_Regular,y=s.P_Premium,mode="markers",name=ch,marker={"size":10,"color":COLORS[ch],"line":{"color":"white","width":1.5}},customdata=s[["T","Quantity"]].to_numpy(),hovertemplate="<b>%{fullData.name}</b><br>Week %{customdata[0]}<br>Regular €%{x:.0f}<br>Premium €%{y:.0f}<br>Quantity %{customdata[1]}<extra></extra>"))
    sc=probabilities(h,pr,pp); pred=max(CHOICES,key=sc.get); fig.add_trace(go.Scatter(x=[pr],y=[pp],mode="markers",name="Scenario",marker={"symbol":"star","size":18,"color":"#172033","line":{"color":"white","width":2}}))
    fig.update_layout(height=500,margin={"l":10,"r":10,"t":20,"b":10},paper_bgcolor="white",plot_bgcolor="white",font={"family":"Inter, Arial, sans-serif","color":"#172033"},xaxis={"title":"Regular price (€)","range":[30,70],"gridcolor":"#e8edf3","zeroline":False,"fixedrange":True},yaxis={"title":"Premium price (€)","range":[55,145],"gridcolor":"#e8edf3","zeroline":False,"fixedrange":True},legend={"orientation":"h","y":1.04,"x":0},hoverlabel={"bgcolor":"white"})
    return fig,sc,pred,(ps,pi),(cs,ci)

def qty(s):
    r=s.loc[s.Choice=="Regular","Regular"]; p=s.loc[s.Choice=="Premium","Premium"]; return float(s.Quantity.mean()),float(r.mean()) if len(r) else 0,float(p.mean()) if len(p) else 0

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--ink:#101828;--muted:#667085;--line:#e5eaf0;--page:#f3f5f7;--blue:#1677ff;--orange:#ff8a1f}
html,body,[class*="css"],.stApp{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important}.stApp,[data-testid="stAppViewContainer"]{background:var(--page);color:var(--ink)}[data-testid="stSidebar"]{display:none}.block-container{max-width:1460px;padding:20px 28px 54px}
.top-shell{background:rgba(255,255,255,.98);border:1px solid var(--line);border-radius:22px;padding:13px 18px;margin-bottom:16px;box-shadow:0 8px 24px rgba(16,24,40,.035)}.brand{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:700;color:#172033}.brand-icon{width:28px;height:28px;border-radius:8px;display:grid;place-items:center;color:white;background:linear-gradient(145deg,#ffb21a,#ff8a1f);font-size:16px}.nav{display:flex;gap:26px;align-items:center;justify-content:center}.nav a{color:#344054;text-decoration:none;font-size:13px}.nav a.active{background:#172033;color:white;padding:9px 16px;border-radius:11px;box-shadow:0 4px 12px rgba(16,24,40,.12)}.profile{width:34px;height:34px;border:1px solid var(--line);border-radius:50%;display:grid;place-items:center;color:#344054}
.scenario-shell{background:#fff;border:1px solid var(--line);border-radius:18px;padding:13px 16px;margin-bottom:26px;box-shadow:0 5px 18px rgba(16,24,40,.025)}.scenario-title{font-size:12px;font-weight:600;color:#344054;margin-bottom:7px}.scenario-help{font-size:11px;color:#98a2b3}.scenario-shell [data-testid="stSlider"]{padding-bottom:0}.scenario-shell [data-testid="stSelectbox"]{padding-bottom:0}.scenario-shell [data-testid="stWidgetLabel"]{font-size:11px!important;color:#667085!important}.scenario-shell [data-baseweb="slider"] div[role="slider"]{background:#1677ff!important;border-color:#1677ff!important}
.eyebrow{display:inline-block;padding:7px 11px;border-radius:999px;background:#edf3f9;color:#4b6380;font-size:11px;font-weight:600;margin-bottom:10px}.hero{margin:4px 0 24px}.hero-copy{max-width:780px}.hero h1{font-size:38px;line-height:1.05;letter-spacing:-1.8px;margin:0 0 12px;color:#101828;font-weight:600}.hero p{margin:0;color:#667085;font-size:14px;line-height:1.6}.card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 5px 20px rgba(16,24,40,.025);height:100%}.card-title{color:#101828;font-size:13px;font-weight:600;margin-bottom:5px}.card-sub{color:#98a2b3;font-size:11px;line-height:1.5}.kpi-value{color:#101828;font-size:26px;font-weight:600;letter-spacing:-1px;margin-top:6px}.insight{background:linear-gradient(145deg,#172033,#223a53);color:#fff;border-radius:18px;padding:20px;min-height:180px;box-shadow:0 14px 30px rgba(23,32,51,.14)}.insight .eyebrow{background:rgba(255,255,255,.10);color:#dbeafe}.insight h2{color:white;font-size:27px;margin:5px 0;letter-spacing:-1px}.insight p{color:#cbd5e1;font-size:12px;margin:0}.prob-row{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.10);font-size:12px}.prob-row:last-child{border-bottom:0}.pill{display:inline-block;padding:5px 9px;border-radius:999px;font-size:10px;font-weight:600;background:#f2f4f7;color:#475467;margin:2px 2px 0 0}.equation{background:#f8fafc;border:1px solid #e7edf3;border-radius:12px;padding:11px 13px;margin-top:9px;color:#172033;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}.info{background:#f5f9ff;border:1px solid #dcecff;border-radius:14px;padding:13px 15px;color:#475467;font-size:11px;line-height:1.5}[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:14px 15px;box-shadow:0 4px 16px rgba(16,24,40,.025)}[data-testid="stMetricLabel"]{font-size:11px!important;color:#667085!important}[data-testid="stMetricValue"]{font-size:25px!important;color:#101828!important}div[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:#fff}[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden}
@media(max-width:900px){.block-container{padding:16px 16px 38px}.nav{gap:13px}.hero h1{font-size:32px}.scenario-shell{padding:12px}.scenario-title{margin-bottom:4px}}
@media(max-width:640px){.block-container{padding:10px 10px 30px}.top-shell{border-radius:18px;padding:12px}.nav{display:none}.brand{font-size:16px}.hero h1{font-size:27px;letter-spacing:-1px}.hero p{font-size:13px}.card{border-radius:15px;padding:14px}.insight{border-radius:15px}[data-testid="stMetric"]{padding:11px 12px}[data-testid="stMetricValue"]{font-size:21px!important}.js-plotly-plot .plotly{width:100%!important}}
</style>
""",unsafe_allow_html=True)

st.markdown("""<div class="top-shell"><div style="display:flex;align-items:center;justify-content:space-between;gap:18px"><div class="brand"><span class="brand-icon">☕</span> Coffee Capsule Lab</div><div class="nav"><a class="active" href="#home">Home</a><a href="#model">Model</a><a href="#data">Data</a><a href="#methodology">Methodology</a><a href="#insights">Insights</a></div><div class="profile">◦</div></div></div><div id="home"></div>""",unsafe_allow_html=True)

st.markdown('<div class="scenario-shell"><div class="scenario-title">Scenario controls</div><div class="scenario-help">Choose a household and test a Regular / Premium price pair. These controls are intentionally separated from the navigation.</div></div>',unsafe_allow_html=True)
sc1,sc2,sc3=st.columns([1.1,1,1])
with sc1: household=st.selectbox("Household",list(HOUSEHOLDS),key="root_household")
with sc2: regular_price=st.slider("Regular price (€)",30,70,BASELINE_R,key="root_regular_price")
with sc3: premium_price=st.slider("Premium price (€)",65,105,BASELINE_P,key="root_premium_price")

fig,scenario,predicted,purchase_boundary,choice_boundary=decision_map(household,regular_price,premium_price)
avg,qr,qp=qty(OBS[OBS.Household==household])

st.markdown(f'<div class="hero"><div class="hero-copy"><div class="eyebrow">Two-stage logistic classifier</div><h1>Coffee Capsules<br>Purchase & Choice Analysis</h1><p>Explore how Regular and Premium prices change the estimated probability of purchasing, choosing Premium, or not purchasing for each household.</p></div></div>',unsafe_allow_html=True)

m1,m2,m3,m4=st.columns(4);m1.metric("Purchase probability",f"{scenario['Buy']:.0%}");m2.metric("P(Regular)",f"{scenario['Regular']:.0%}");m3.metric("P(Premium)",f"{scenario['Premium']:.0%}");m4.metric("P(No Purchase)",f"{scenario['No Purchase']:.0%}")

st.markdown('<div id="model"></div>',unsafe_allow_html=True);left,right=st.columns([1.65,.8],gap="medium")
with left:
    st.markdown('<div class="card"><div class="card-title">Decision boundaries</div><div class="card-sub">Observed weekly choices and the 50% prediction boundaries implied by the two logistic stages.</div></div>',unsafe_allow_html=True);st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False,"responsive":True})
with right:
    st.markdown(f'<div class="insight"><div class="eyebrow">Scenario result</div><h2>{predicted}</h2><p>At Regular €{regular_price} and Premium €{premium_price}</p><div style="height:12px"></div><div class="prob-row"><span>Regular</span><b>{scenario["Regular"]:.1%}</b></div><div class="prob-row"><span>Premium</span><b>{scenario["Premium"]:.1%}</b></div><div class="prob-row"><span>No Purchase</span><b>{scenario["No Purchase"]:.1%}</b></div></div>',unsafe_allow_html=True);st.write("");st.markdown(f'<div class="card"><div class="card-title">Demand snapshot</div><div class="card-sub">Observed quantity across the 11 weeks</div><div class="kpi-value">{avg:.2f}</div><div class="card-sub">capsules / week</div><div style="height:8px"></div><span class="pill">Regular when purchased&nbsp; {qr:.2f}</span><span class="pill">Premium when purchased&nbsp; {qp:.2f}</span></div>',unsafe_allow_html=True)

st.markdown('<div id="insights"></div>',unsafe_allow_html=True);i1,i2,i3=st.columns(3)
with i1: st.markdown('<div class="card"><div class="card-title">Purchase stage</div><div class="card-sub">Estimates P(Buy) using Regular price, Premium price, and household indicators.</div><div class="equation">logit(P(Buy)) = β₀ + βᵣzᵣ + βₚzₚ + household effects</div></div>',unsafe_allow_html=True)
with i2: st.markdown('<div class="card"><div class="card-title">Choice stage</div><div class="card-sub">Among observed buyers, estimates P(Premium | Buy).</div><div class="equation">logit(P(Premium | Buy)) = γ₀ + γᵣzᵣ + γₚzₚ + household effects</div></div>',unsafe_allow_html=True)
with i3: st.markdown(f'<div class="card"><div class="card-title">Current household</div><div class="card-sub">{household} · 11 weekly observations</div><div class="kpi-value">{max(scenario["Regular"],scenario["Premium"],scenario["No Purchase"]):.0%}</div><div class="card-sub">highest predicted choice probability</div></div>',unsafe_allow_html=True)

st.markdown('<div id="methodology"></div>',unsafe_allow_html=True);st.subheader("Model & methodology")
with st.expander("How the two-stage model works"): st.markdown(f"""**Stage 1 — Purchase.** A pooled logistic regression estimates the probability that the household purchases anything.

**Stage 2 — Product choice.** For observations where a purchase is observed, a second logistic regression estimates the probability of Premium rather than Regular.

The final probabilities are:
- **P(No Purchase) = 1 − P(Buy)**
- **P(Regular) = P(Buy) × [1 − P(Premium | Buy)]**
- **P(Premium) = P(Buy) × P(Premium | Buy)**

For the current scenario, the model predicts **{scenario['Regular']:.1%} Regular, {scenario['Premium']:.1%} Premium, and {scenario['No Purchase']:.1%} No Purchase**.""")
with st.expander("How should I read the boundaries?"):
    ps,pi=purchase_boundary;cs,ci=choice_boundary
    st.markdown(f"""The dashed lines are **50% prediction contours**. They are not literal willingness-to-pay curves.

**Purchase boundary**  
`P_P = {ps:.2f} × P_R + {pi:.1f}`

**Regular / Premium boundary**  
`P_P = {cs:.2f} × P_R + {ci:.1f}`

The background region shows which choice the model predicts as most likely at each price combination.""")
with st.expander("Important interpretation note"): st.markdown("This is an exploratory classification model rather than a structural demand or causal price-elasticity model. The boundary lines are useful for visualizing model predictions, but should not be presented as validated willingness-to-pay thresholds.")

st.markdown('<div id="data"></div>',unsafe_allow_html=True);st.subheader("Observed weeks");sample=OBS[OBS.Household==household][["T","P_Regular","P_Premium","Choice","Quantity"]].copy();sample.columns=["Week","Regular price","Premium price","Observed choice","Quantity"];st.dataframe(sample,use_container_width=True,hide_index=True);st.caption("33 household-week observations · 3 households · exploratory research model")
