import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Coffee Capsule Demand", page_icon="☕", layout="wide", initial_sidebar_state="expanded")
RAW=pd.read_csv("coffee_capsules_data.csv")
HOUSEHOLDS={"Household 1":("HH1_Regular","HH1_Premium"),"Household 2":("HH2_Regular","HH2_Premium"),"Household 3":("HH3_Regular","HH3_Premium")}
CHOICES=["Regular","Premium","No Purchase"]
COLORS={"Regular":"#18794E","Premium":"#D99A19","No Purchase":"#667085"}

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
purchase_model=LogisticRegression(C=1,max_iter=5000).fit(features(OBS),(OBS.Choice!="No Purchase").astype(int))
BUYERS=OBS[OBS.Choice!="No Purchase"]; choice_model=LogisticRegression(C=1,max_iter=5000).fit(features(BUYERS),(BUYERS.Choice=="Premium").astype(int))

def probabilities(h,pr,pp):
    x=pd.DataFrame({"Household":[h],"P_Regular":[pr],"P_Premium":[pp]}); buy=purchase_model.predict_proba(features(x))[0,1]; prem=choice_model.predict_proba(features(x))[0,1]
    return {"No Purchase":1-buy,"Regular":buy*(1-prem),"Premium":buy*prem,"Buy":buy,"Premium | Buy":prem}

def boundary(model,h):
    c=model.coef_[0]; he=c[2] if h=="Household 2" else c[3] if h=="Household 3" else 0
    if abs(c[1])<1e-12:return np.nan,np.nan
    sz=-c[0]/c[1]; iz=-(model.intercept_[0]+he)/c[1]; return float(sz*PP_STD/PR_STD),float(PP_MEAN+PP_STD*iz-(sz*PP_STD/PR_STD)*PR_MEAN)

def decision_map(h,pr,pp):
    ps,pi=boundary(purchase_model,h); cs,ci=boundary(choice_model,h); xs=np.linspace(30,70,180); ys=np.linspace(55,145,180); xx,yy=np.meshgrid(xs,ys)
    grid=pd.DataFrame({"Household":h,"P_Regular":xx.ravel(),"P_Premium":yy.ravel()}); buy=purchase_model.predict_proba(features(grid))[:,1].reshape(xx.shape); prem=choice_model.predict_proba(features(grid))[:,1].reshape(xx.shape); z=np.where(buy<.5,2,np.where(prem>=.5,1,0))
    fig=go.Figure(go.Heatmap(x=xs,y=ys,z=z,zmin=0,zmax=2,colorscale=[[0,"rgba(24,121,78,.10)"],[.33,"rgba(24,121,78,.10)"],[.34,"rgba(217,154,25,.10)"],[.66,"rgba(217,154,25,.10)"],[.67,"rgba(102,112,133,.10)"],[1,"rgba(102,112,133,.10)"]],showscale=False,hoverinfo="skip"))
    for line,name,dash in [(ps*xs+pi,"Buy / No Purchase","dash"),(cs*xs+ci,"Regular / Premium","solid")]:
        mask=np.isfinite(line)&(line>=ys.min())&(line<=ys.max())
        if mask.any():fig.add_trace(go.Scatter(x=xs[mask],y=line[mask],mode="lines",name=name,line=dict(color="#172033",width=3,dash=dash)))
    hd=OBS[OBS.Household==h]
    for ch in CHOICES:
        s=hd[hd.Choice==ch]
        if not s.empty:fig.add_trace(go.Scatter(x=s.P_Regular,y=s.P_Premium,mode="markers",name=ch,marker=dict(size=12,color=COLORS[ch],line=dict(color="white",width=1.5)),customdata=s[["T","Quantity"]].to_numpy(),hovertemplate="<b>%{fullData.name}</b><br>Week %{customdata[0]}<br>P_R=%{x:.0f}<br>P_P=%{y:.0f}<br>Quantity=%{customdata[1]}<extra></extra>"))
    sc=probabilities(h,pr,pp); pred=max(CHOICES,key=sc.get); fig.add_trace(go.Scatter(x=[pr],y=[pp],mode="markers",name="Scenario",marker=dict(symbol="star",size=20,color="#172033",line=dict(color="white",width=2)),hovertemplate=f"<b>Scenario</b><br>P_R={pr:.0f}<br>P_P={pp:.0f}<br>Most likely: {pred}<extra></extra>"))
    fig.update_layout(height=540,margin=dict(l=5,r=5,t=15,b=5),paper_bgcolor="white",plot_bgcolor="white",xaxis=dict(title="Regular price",range=[30,70],gridcolor="#EAECF0",zeroline=False),yaxis=dict(title="Premium price",range=[55,145],gridcolor="#EAECF0",zeroline=False),legend=dict(orientation="h",y=1.02,x=0),hoverlabel=dict(bgcolor="white")); return fig,sc,(ps,pi),(cs,ci)

def qty(s):
    r=s.loc[s.Choice=="Regular","Regular"]; p=s.loc[s.Choice=="Premium","Premium"]; return float(s.Quantity.mean()),float(r.mean()) if len(r) else 0,float(p.mean()) if len(p) else 0

st.markdown("""<style>.block-container{max-width:1400px;padding:1.2rem 2rem 3rem;background:#f6f7f5}[data-testid="stAppViewContainer"]{background:#f6f7f5}section[data-testid="stSidebar"]{background:#fbfcfa;border-right:1px solid #e5e7eb}.brand{font-size:1.12rem;font-weight:750;color:#16352a;padding:8px 0 22px}.navlabel{font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:#98a2b3;margin:18px 0 8px}.navitem{padding:9px 11px;border-radius:10px;color:#475467;font-size:.9rem}.active{background:#e5f4ec;color:#176b48;font-weight:700}.topbar{display:flex;justify-content:space-between;align-items:center;background:white;border:1px solid #e5e7eb;border-radius:18px;padding:11px 16px;margin-bottom:18px}.search{background:#f6f7f8;border-radius:11px;padding:9px 14px;color:#98a2b3;font-size:.85rem;width:45%}.eyebrow{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#667085;font-weight:700}.subtitle{color:#667085;font-size:.92rem}.card{background:#fff;border:1px solid #e5e7eb;border-radius:18px;padding:18px 20px;height:100%;box-shadow:0 2px 8px rgba(16,24,40,.025)}.card-title{font-weight:700;color:#172033;font-size:.96rem;margin-bottom:8px}.mini{color:#667085;font-size:.78rem}.insight{background:#163f30;color:white;border-radius:18px;padding:20px;height:100%}.insight .mini{color:#c8ddd4}.insight h3{margin:5px 0 8px;color:white}[data-testid="stMetric"]{background:white;border:1px solid #e5e7eb;border-radius:15px;padding:13px 15px}div[data-testid="stExpander"]{border:1px solid #e5e7eb;border-radius:14px;background:white}</style>""",unsafe_allow_html=True)
with st.sidebar:
    st.markdown('<div class="brand">☕ Coffee Capsule Lab</div>',unsafe_allow_html=True); st.markdown('<div class="navlabel">Workspace</div><div class="navitem active">▦ Dashboard</div><div class="navitem">◌ Experiment</div><div class="navitem">⌁ Model</div><div class="navitem">◫ Observations</div><div class="navlabel">Explore</div><div class="navitem">◉ About</div>',unsafe_allow_html=True); st.divider(); household=st.selectbox("Household",list(HOUSEHOLDS)); st.markdown('<div class="navlabel">Scenario</div>',unsafe_allow_html=True); p_r=st.slider("Regular price",30,70,42); p_p=st.slider("Premium price",65,105,93); st.caption("33 household-week observations · 3 households")
fig,sc,pb,cb=decision_map(household,p_r,p_p); pred=max(CHOICES,key=sc.get); avg,qr,qp=qty(OBS[OBS.Household==household])
st.markdown('<div class="topbar"><div class="search">⌕ &nbsp; Explore the experiment</div><div class="mini">Independent price-choice research tool</div></div>',unsafe_allow_html=True); st.markdown('<div class="eyebrow">Coffee capsule experiment</div>',unsafe_allow_html=True); st.title("Household demand dashboard"); st.markdown('<div class="subtitle">See how Regular and Premium prices change the estimated probability of buying, switching, or not purchasing.</div>',unsafe_allow_html=True); st.write("")
a,b,c,d=st.columns(4); a.metric("Most likely choice",pred); b.metric("P(Regular)",f"{sc['Regular']:.0%}"); c.metric("P(Premium)",f"{sc['Premium']:.0%}"); d.metric("P(No Purchase)",f"{sc['No Purchase']:.0%}"); st.write("")
left,right=st.columns([1.65,.8])
with left: st.markdown('<div class="card"><div class="card-title">Decision map</div><div class="mini">Observed choices, model boundaries and your selected price scenario.</div></div>',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True)
with right: st.markdown(f'<div class="insight"><div class="eyebrow" style="color:#9bc9b5">Scenario result</div><h3>{pred}</h3><div class="mini">Regular = {p_r} · Premium = {p_p}</div><hr style="border-color:#3b6254"><b>Estimated probabilities</b><br><br>Regular &nbsp; <b>{sc["Regular"]:.1%}</b><br>Premium &nbsp; <b>{sc["Premium"]:.1%}</b><br>No Purchase &nbsp; <b>{sc["No Purchase"]:.1%}</b></div>',unsafe_allow_html=True); st.write(""); st.markdown(f'<div class="card"><div class="card-title">Demand snapshot</div><div class="mini">Observed average, all weeks included</div><h2>{avg:.2f}</h2><div class="mini">capsules / week</div><br>Regular when purchased: <b>{qr:.2f}</b><br>Premium when purchased: <b>{qp:.2f}</b></div>',unsafe_allow_html=True)
st.write(""); x,y,z=st.columns(3)
with x: st.markdown('<div class="card"><div class="card-title">Buy boundary</div><h3>50%</h3><div class="mini">Dashed line: Buy vs No Purchase.</div></div>',unsafe_allow_html=True)
with y: st.markdown('<div class="card"><div class="card-title">Switch boundary</div><h3>50%</h3><div class="mini">Solid line: Regular vs Premium given purchase.</div></div>',unsafe_allow_html=True)
with z: st.markdown('<div class="card"><div class="card-title">Model design</div><h3>2-stage logistic</h3><div class="mini">Pooled model with household indicators and 33 observations.</div></div>',unsafe_allow_html=True)
st.subheader("Understand the model")
with st.expander("How the dashboard works"): st.markdown(f"""**Stage 1 — Buy vs No Purchase.** A pooled logistic regression estimates P(Buy) from both prices and household indicators.

**Stage 2 — Regular vs Premium.** Among observed buyers, a second logistic regression estimates P(Premium | Buy).

The three final probabilities are P(No Purchase)=1−P(Buy), P(Regular)=P(Buy)×[1−P(Premium|Buy)], and P(Premium)=P(Buy)×P(Premium|Buy).

Current scenario: **{sc['Regular']:.1%} Regular · {sc['Premium']:.1%} Premium · {sc['No Purchase']:.1%} No Purchase**.""")
with st.expander("Where do the boundary equations come from?"): st.markdown(f"""Each boundary is the **50% contour** of its binary logistic model. Setting log-odds to zero and converting the standardised price variables back to original price units gives a straight line.

**Buy / No Purchase:** `P_P = {pb[0]:.2f} × P_R + {pb[1]:.1f}`

**Regular / Premium | Buy:** `P_P = {cb[0]:.2f} × P_R + {cb[1]:.1f}`""")
st.subheader("Observed weeks"); sample=OBS[OBS.Household==household][["T","P_Regular","P_Premium","Choice","Quantity"]].copy(); sample.columns=["Week","Regular price","Premium price","Choice","Quantity"]; st.dataframe(sample,use_container_width=True,hide_index=True); st.caption("Exploratory model: useful for understanding this experiment, but not a causal elasticity or validated willingness-to-pay model.")
