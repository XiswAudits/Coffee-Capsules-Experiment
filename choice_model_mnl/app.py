import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="Coffee Capsule Choice Model",page_icon="☕",layout="wide",initial_sidebar_state="expanded")
RAW=pd.read_csv("coffee_capsules_data.csv")
HOUSEHOLDS={"Household 1":("HH1_Regular","HH1_Premium"),"Household 2":("HH2_Regular","HH2_Premium"),"Household 3":("HH3_Regular","HH3_Premium")}
CHOICES=["Regular","Premium","No Purchase"]
COLORS={"Regular":"#18794E","Premium":"#D99A19","No Purchase":"#667085"}

def build_obs():
    rows=[]
    for h,(rc,pc) in HOUSEHOLDS.items():
        for _,r in RAW.iterrows():
            rq,pq=int(r[rc]),int(r[pc]); c="Regular" if rq>0 else "Premium" if pq>0 else "No Purchase"
            rows.append({"Household":h,"T":int(r["T"]),"P_Regular":float(r["P_Regular"]),"P_Premium":float(r["P_Premium"]),"Regular":rq,"Premium":pq,"Choice":c,"Quantity":rq+pq})
    return pd.DataFrame(rows)
OBS=build_obs(); PR_MEAN=OBS.P_Regular.mean(); PR_STD=OBS.P_Regular.std(ddof=0); PP_MEAN=OBS.P_Premium.mean(); PP_STD=OBS.P_Premium.std(ddof=0)
BOUNDS=[(-20.,20.)]*4; PARAMS=["ASC_Regular","ASC_Premium","Beta_Regular_Price","Beta_Premium_Price"]

def softmax(u):
    u=u-np.max(u,axis=1,keepdims=True); e=np.exp(u); return e/e.sum(axis=1,keepdims=True)

def utility(p,pr,pp):
    ar,ap,br,bp=p; zr=(np.asarray(pr)-PR_MEAN)/PR_STD; zp=(np.asarray(pp)-PP_MEAN)/PP_STD
    return np.column_stack([ar+br*zr,ap+bp*zp,np.zeros_like(zr)])

def fit(h):
    s=OBS[OBS.Household==h].reset_index(drop=True); y=s.Choice.map({"Regular":0,"Premium":1,"No Purchase":2}).to_numpy()
    def nll(p):
        q=softmax(utility(p,s.P_Regular,s.P_Premium)); return float(-np.log(np.clip(q[np.arange(len(y)),y],1e-15,1)).sum())
    starts=[np.array([0,0,-1,-1.]),np.array([1,0,-1,-1.]),np.array([0,1,-1,-1.])]
    rs=[minimize(nll,x,method="L-BFGS-B",bounds=BOUNDS) for x in starts]; res=min(rs,key=lambda x:x.fun); p=res.x; q=softmax(utility(p,s.P_Regular,s.P_Premium))
    at={PARAMS[i]:bool(abs(p[i]-BOUNDS[i][0])<1e-5 or abs(p[i]-BOUNDS[i][1])<1e-5) for i in range(4)}
    return {"sample":s,"params":p,"probs":q,"predicted":q.argmax(1),"observed":y,"at_bound":at,"nll":float(res.fun)}
MODELS={h:fit(h) for h in HOUSEHOLDS}

def probabilities(m,pr,pp):
    q=softmax(utility(m["params"],[pr],[pp]))[0]; return dict(zip(CHOICES,q.astype(float)))

def boundaries(m):
    ar,ap,br,bp=m["params"]; rn=PR_MEAN-ar*PR_STD/br if abs(br)>1e-12 else np.nan; pn=PP_MEAN-ap*PP_STD/bp if abs(bp)>1e-12 else np.nan
    if abs(bp)>1e-12: slope=br*PP_STD/(bp*PR_STD); intercept=PP_MEAN+(ar-ap)*PP_STD/bp-slope*PR_MEAN
    else: slope=intercept=np.nan
    return rn,pn,slope,intercept

def chart(h,pr,pp):
    m=MODELS[h]; s=m["sample"]; xs=np.linspace(30,70,180); ys=np.linspace(55,145,180); xx,yy=np.meshgrid(xs,ys); q=softmax(utility(m["params"],xx.ravel(),yy.ravel())); z=q.argmax(1).reshape(xx.shape)
    fig=go.Figure(go.Heatmap(x=xs,y=ys,z=z,zmin=0,zmax=2,colorscale=[[0,"rgba(24,121,78,.10)"],[.33,"rgba(24,121,78,.10)"],[.34,"rgba(217,154,25,.10)"],[.66,"rgba(217,154,25,.10)"],[.67,"rgba(102,112,133,.10)"],[1,"rgba(102,112,133,.10)"]],showscale=False,hoverinfo="skip"))
    rn,pn,sl,it=boundaries(m)
    if np.isfinite(rn) and 30<=rn<=70: fig.add_vline(x=rn,line_width=2,line_dash="dash",line_color="#172033")
    if np.isfinite(pn) and 55<=pn<=145: fig.add_hline(y=pn,line_width=2,line_dash="dash",line_color="#172033")
    if np.isfinite(sl) and np.isfinite(it):
        line=sl*xs+it; mask=(line>=55)&(line<=145)
        if mask.any(): fig.add_trace(go.Scatter(x=xs[mask],y=line[mask],mode="lines",name="Regular = Premium",line=dict(color="#172033",width=3)))
    for c in CHOICES:
        d=s[s.Choice==c]
        if not d.empty: fig.add_trace(go.Scatter(x=d.P_Regular,y=d.P_Premium,mode="markers",name=c,marker=dict(size=12,color=COLORS[c],line=dict(color="white",width=1.5)),customdata=d[["T","Quantity"]].to_numpy(),hovertemplate="<b>%{fullData.name}</b><br>Week %{customdata[0]}<br>Regular=%{x:.0f}<br>Premium=%{y:.0f}<br>Quantity=%{customdata[1]}<extra></extra>"))
    sc=probabilities(m,pr,pp); pred=max(CHOICES,key=sc.get); fig.add_trace(go.Scatter(x=[pr],y=[pp],mode="markers",name="Scenario",marker=dict(symbol="star",size=20,color="#172033",line=dict(color="white",width=2))))
    fig.update_layout(height=535,margin=dict(l=5,r=5,t=10,b=5),paper_bgcolor="white",plot_bgcolor="white",xaxis=dict(title="Regular price",range=[30,70],gridcolor="#EAECF0",zeroline=False),yaxis=dict(title="Premium price",range=[55,145],gridcolor="#EAECF0",zeroline=False),legend=dict(orientation="h",y=1.02,x=0),hoverlabel=dict(bgcolor="white")); return fig,sc,pred

def qstats(s):
    r=s.loc[s.Choice=="Regular","Regular"]; p=s.loc[s.Choice=="Premium","Premium"]; return float(s.Quantity.mean()),float(r.mean()) if len(r) else 0.,float(p.mean()) if len(p) else 0.

st.markdown("""<style>
.block-container{max-width:1400px;padding:1.2rem 2rem 3rem;background:#f6f7f5}[data-testid="stAppViewContainer"]{background:#f6f7f5}section[data-testid="stSidebar"]{background:#fbfcfa;border-right:1px solid #e5e7eb}.brand{font-size:1.12rem;font-weight:750;color:#16352a;padding:8px 0 22px}.navlabel{font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:#98a2b3;margin:18px 0 8px}.navitem{padding:9px 11px;border-radius:10px;color:#475467;font-size:.9rem}.active{background:#e5f4ec;color:#176b48;font-weight:700}.topbar{display:flex;justify-content:space-between;align-items:center;background:white;border:1px solid #e5e7eb;border-radius:18px;padding:11px 16px;margin-bottom:18px}.search{background:#f6f7f8;border-radius:11px;padding:9px 14px;color:#98a2b3;font-size:.85rem;width:45%}.eyebrow{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#667085;font-weight:700}.subtitle{color:#667085;font-size:.92rem}.card{background:#fff;border:1px solid #e5e7eb;border-radius:18px;padding:18px 20px;height:100%;box-shadow:0 2px 8px rgba(16,24,40,.025)}.card-title{font-weight:700;color:#172033;font-size:.96rem;margin-bottom:8px}.mini{color:#667085;font-size:.78rem}.insight{background:#163f30;color:white;border-radius:18px;padding:20px;height:100%}.insight .mini{color:#c8ddd4}.insight h3{margin:5px 0 8px;color:white}.eq{background:#f8fafc;border:1px solid #e5e7eb;border-radius:12px;padding:12px 15px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#172033;margin:8px 0}[data-testid="stMetric"]{background:white;border:1px solid #e5e7eb;border-radius:15px;padding:13px 15px}div[data-testid="stExpander"]{border:1px solid #e5e7eb;border-radius:14px;background:white}
</style>""",unsafe_allow_html=True)
with st.sidebar:
    st.markdown('<div class="brand">☕ Choice Model Lab</div>',unsafe_allow_html=True)
    st.markdown('<div class="navlabel">Workspace</div><div class="navitem active">▦ Dashboard</div><div class="navitem">◌ Experiment</div><div class="navitem">⌁ Model</div><div class="navitem">◫ Observations</div><div class="navlabel">Explore</div><div class="navitem">◉ Methodology</div>',unsafe_allow_html=True)
    st.divider(); household=st.selectbox("Household",list(HOUSEHOLDS)); st.markdown('<div class="navlabel">Scenario</div>',unsafe_allow_html=True); pr=st.slider("Regular price",30,70,42); pp=st.slider("Premium price",65,105,93); st.caption("33 observations · 11 weeks · 3 independent MNLs")
fig,sc,pred=chart(household,pr,pp); m=MODELS[household]; avg,qr,qp=qstats(m["sample"]); rn,pn,sl,it=boundaries(m); ar,ap,br,bp=m["params"]
st.markdown('<div class="topbar"><div class="search">⌕ &nbsp; Explore the experiment</div><div class="mini">Independent household choice research</div></div>',unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Discrete choice demand experiment</div>',unsafe_allow_html=True); st.title("Household choice dashboard"); st.markdown('<div class="subtitle">An independent multinomial logit model for each household. Explore how price changes the probability of choosing Regular, Premium, or No Purchase.</div>',unsafe_allow_html=True); st.write("")
a,b,c,d=st.columns(4); a.metric("Most likely choice",pred); b.metric("P(Regular)",f"{sc['Regular']:.0%}"); c.metric("P(Premium)",f"{sc['Premium']:.0%}"); d.metric("P(No Purchase)",f"{sc['No Purchase']:.0%}")
left,right=st.columns([1.65,.8])
with left:
    st.markdown('<div class="card"><div class="card-title">Decision map</div><div class="mini">Observed choices, equal-utility boundaries and the current scenario.</div></div>',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True)
with right:
    st.markdown(f'<div class="insight"><div class="eyebrow" style="color:#9bc9b5">Scenario result</div><h3>{pred}</h3><div class="mini">Regular = {pr} · Premium = {pp}</div><hr style="border-color:#3b6254"><b>Estimated probabilities</b><br><br>Regular &nbsp; <b>{sc["Regular"]:.1%}</b><br>Premium &nbsp; <b>{sc["Premium"]:.1%}</b><br>No Purchase &nbsp; <b>{sc["No Purchase"]:.1%}</b></div>',unsafe_allow_html=True); st.write(""); st.markdown(f'<div class="card"><div class="card-title">Demand snapshot</div><div class="mini">Observed average across all 11 weeks</div><h2>{avg:.2f}</h2><div class="mini">capsules / week</div><br>Regular when purchased: <b>{qr:.2f}</b><br>Premium when purchased: <b>{qp:.2f}</b></div>',unsafe_allow_html=True)
st.write(""); x,y,z=st.columns(3)
with x: st.markdown('<div class="card"><div class="card-title">Utility</div><div class="mini">Each household has its own fitted utility.</div><div class="eq">V_R = ASC_R + β_R · z_R</div><div class="eq">V_P = ASC_P + β_P · z_P</div><div class="eq">V_N = 0</div></div>',unsafe_allow_html=True)
with y: st.markdown(f'<div class="card"><div class="card-title">Selected household</div><div class="mini">{household}</div><h3>{ar:.2f} / {ap:.2f}</h3><div class="mini">ASCs: Regular / Premium</div><h3>{br:.2f} / {bp:.2f}</h3><div class="mini">Price coefficients: Regular / Premium</div></div>',unsafe_allow_html=True)
with z: st.markdown(f'<div class="card"><div class="card-title">Choice boundaries</div><div class="mini">Equal-utility thresholds in original price units.</div><br>Regular = Premium<br><b>P_P = {sl:.2f} × P_R + {it:.1f}</b><br><br>Regular = No Purchase<br><b>P_R ≈ {rn:.1f}</b><br><br>Premium = No Purchase<br><b>P_P ≈ {pn:.1f}</b></div>',unsafe_allow_html=True)

st.subheader("Understand the model")
with st.expander("Where do the numbers come from?"):
    st.markdown(f"""The model is estimated **separately for {household}** using that household's 11 observed weekly choices.

**Price scaling** is calculated from all 33 observations and is used only for numerical stability:
- `z_R = (P_R − {PR_MEAN:.2f}) / {PR_STD:.2f}`
- `z_P = (P_P − {PP_MEAN:.2f}) / {PP_STD:.2f}`

The four household-specific parameters are estimated by **maximum likelihood**. The optimiser searches for the values that give the highest probability to the choices actually observed in those 11 weeks.

**No Purchase = 0** is the reference-alternative convention. It fixes the utility scale, so the two ASCs are interpreted relative to No Purchase.

**Boundaries** come from setting two utilities equal. For example, Regular = Premium means `V_R = V_P`; rearranging that equation gives the straight line shown on the map.""")
with st.expander("Why MNL? Why independent households?"):
    st.markdown("""The MNL treats **Regular, Premium, and No Purchase as three competing alternatives in one Random Utility Model**. This is different from the previous two-stage classifier.

This version is intentionally **independent by household**: every household gets its own utility parameters. That makes the dashboard directly answer how a particular household responds to the price pair.

The trade-off is sample size: each household has only 11 observations, and some alternatives are never observed. The model therefore uses finite parameter bounds and flags estimates that reach them. These are exploratory estimates, not precise causal elasticity or willingness-to-pay measurements.""")
with st.expander("How should I read the decision map?"):
    st.markdown("""- **Dots** are actual observed household-week choices.
- **Background** is the alternative with the highest estimated utility at each price combination.
- **Dashed vertical/horizontal lines** are product vs No Purchase equal-utility thresholds when estimable inside the chart.
- **Solid line** is the Regular vs Premium equal-utility boundary.
- **Star** is the scenario being tested.

The MNL produces probabilities, so these boundaries should not be interpreted as hard causal willingness-to-pay thresholds.""")
st.subheader("Observed weeks"); d=m["sample"][["T","P_Regular","P_Premium","Choice","Quantity"]].copy(); d.columns=["Week","Regular price","Premium price","Observed choice","Quantity"]; st.dataframe(d,use_container_width=True,hide_index=True)
if any(m["at_bound"].values()):
    hit=[k for k,v in m["at_bound"].items() if v]; st.warning("Numerical diagnostic: "+", ".join(hit)+" reached the ±20 optimisation bound. Treat that estimate as weakly identified rather than as a precise economic effect.")
st.caption("Exploratory research tool · 11 observations per household · independent household MNL")
