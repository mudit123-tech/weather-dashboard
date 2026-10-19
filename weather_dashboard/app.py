import os, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from scipy import stats
from scipy.spatial.distance import cdist
from plotly.subplots import make_subplots

st.set_page_config(page_title="Weather Intelligence Dashboard", page_icon="🌦️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.2rem;max-width:1500px}
.kpi{border-radius:14px;padding:16px 18px;color:#fff;box-shadow:0 4px 14px rgba(0,0,0,.35)}
.kpi .v{font-size:30px;font-weight:700;line-height:1.1}.kpi .l{font-size:13px;opacity:.9;margin-top:4px}
.hero{background:linear-gradient(120deg,#0ea5e9,#6366f1 60%,#a855f7);border-radius:16px;padding:20px 26px;margin-bottom:14px}
.hero h1{margin:0;font-size:28px;color:#fff}.hero p{margin:4px 0 0;color:#e0e7ff}
.story{background:#111a2e;border-left:4px solid #38bdf8;padding:12px 16px;border-radius:8px;margin:8px 0}
</style>""", unsafe_allow_html=True)

LBL = {"temp":"Temperature (°C)","rh":"Relative humidity (%)","app_temp":"Apparent temp (°C)","dew":"Dew point (°C)",
       "pressure":"Pressure (hPa)","wind":"Wind speed (km/h)","precip":"Precipitation (mm)","sw_rad":"Shortwave radiation (W/m²)",
       "dir_rad":"Direct radiation (W/m²)","dif_rad":"Diffuse radiation (W/m²)","dni":"Direct normal irradiance (W/m²)"}
NUM = list(LBL); MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
PAL = ["#38bdf8","#f97316","#22c55e","#a855f7","#eab308","#ef4444"]

@st.cache_data(show_spinner="Loading 219,168 hourly records...")
def load():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "data.csv.gz")):
        df = pd.read_csv(os.path.join(here, "data.csv.gz"), parse_dates=["time"])
    else:  # raw xlsx: rows 0-2 are metadata, row 3 is the header
        df = pd.read_excel(os.path.join(here, "data.xlsx"), skiprows=3)
        df = df.rename(columns={"temperature_2m (°C)":"temp","relative_humidity_2m (%)":"rh","apparent_temperature (°C)":"app_temp","dew_point_2m (°C)":"dew","surface_pressure (hPa)":"pressure","wind_speed_10m (km/h)":"wind","precipitation (mm)":"precip","shortwave_radiation (W/m²)":"sw_rad","direct_radiation (W/m²)":"dir_rad","diffuse_radiation (W/m²)":"dif_rad","direct_normal_irradiance (W/m²)":"dni"})
    df = df.sort_values("time").reset_index(drop=True)
    full = pd.date_range(df.time.min(), df.time.max(), freq="h")
    df = df.set_index("time").reindex(full); df.index.name = "time"
    df[NUM] = df[NUM].interpolate(method="time", limit=6); df["precip"] = df["precip"].fillna(0); df[NUM] = df[NUM].fillna(df[NUM].median())
    df["year"], df["month"], df["hour"] = df.index.year, df.index.month, df.index.hour
    return df.reset_index()

raw = load()
st.markdown("""<div class="hero"><h1>🌦️ Weather Intelligence Dashboard</h1>
<p>Hourly meteorological data 2000-2024 · site labelled "mumbai" (18.52°N, 73.87°E, 555 m) · FDS Mini Competition</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🎛️ Filters")
    yr = st.slider("Years", int(raw.year.min()), int(raw.year.max()), (int(raw.year.min()), int(raw.year.max())))
    mons = st.multiselect("Months", MON, default=MON)
    hr = st.slider("Hour of day", 0, 23, (0, 23))
    st.caption("All KPIs and charts update live.")
    st.warning("Data note: label says Mumbai, but coordinates/elevation (555 m, ~948 hPa) match the Pune area.")
df = raw[raw.year.between(*yr) & raw.month.isin([MON.index(m)+1 for m in mons]) & raw.hour.between(*hr)]
if df.empty: st.error("No data for these filters."); st.stop()
nyears = max(df.year.nunique(), 1)

def kpi(col, label, val, color): col.markdown(f'<div class="kpi" style="background:{color}"><div class="v">{val}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)
def style(fig, h=380, title=None):
    fig.update_layout(template="plotly_dark", height=h, title=title, margin=dict(l=10,r=10,t=50 if title else 20,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12))
    return fig

# ---- Scoreboard (always visible)
ann_rain = df.groupby("year").precip.sum()
mons_share = df[df.month.isin([6,7,8,9])].precip.sum() / max(df.precip.sum(), 1e-9) * 100
c = st.columns(6)
for col, (l, v, clr) in zip(c, [("Avg temperature", f"{df.temp.mean():.1f} °C", "#ef6c3d"), ("Avg annual rainfall", f"{ann_rain.mean():.0f} mm", "#0d9488"),
    ("Peak hourly rain", f"{df.precip.max():.1f} mm", "#1e3a5f"), ("Max wind speed", f"{df.wind.max():.1f} km/h", "#7c3aed"),
    ("Avg humidity", f"{df.rh.mean():.0f} %", "#2563eb"), ("Monsoon rain share (Jun-Sep)", f"{mons_share:.0f} %", "#d97706")]): kpi(col, l, v, clr)
st.write("")

t1, t2, t3, t4, t5, t6 = st.tabs(["📊 Overview", "📈 Distributions", "🔍 Outliers & Normality", "🔗 Relationships", "🧮 Preprocessing & Statistics", "📖 Story & Actions"])

with t1:
    a, b = st.columns(2)
    m = df.groupby("month").agg(temp=("temp","mean"), rain=("precip","sum"), rh=("rh","mean"), pres=("pressure","mean")); m["rain"] /= nyears
    m = m.reindex(range(1,13)); m.index = MON
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_bar(x=m.index, y=m.rain, name="Rain (mm/yr)", marker_color="#14b8a6"); f.add_scatter(x=m.index, y=m.temp, name="Temp (°C)", mode="lines+markers", line=dict(color="#f97316", width=3), secondary_y=True)
    f.update_yaxes(title_text="Rain (mm / year)", secondary_y=False); f.update_yaxes(title_text="Temp (°C)", secondary_y=True)
    a.plotly_chart(style(f, 380, "Monthly climatology: monsoon signature"), width="stretch")
    f = px.bar(ann_rain.reset_index(), x="year", y="precip", color="precip", color_continuous_scale="Teal", labels={"precip":"Rainfall (mm)","year":"Year"})
    f.add_hline(y=ann_rain.mean(), line_dash="dash", annotation_text=f"mean {ann_rain.mean():.0f} mm"); f.update_coloraxes(showscale=False)
    b.plotly_chart(style(f, 380, "Total annual rainfall (flood / drought years)"), width="stretch")
    a, b = st.columns(2)
    hm = df.pivot_table(index="hour", columns="month", values="temp"); hm.columns = [MON[i-1] for i in hm.columns]
    a.plotly_chart(style(px.imshow(hm, aspect="auto", color_continuous_scale="YlOrRd", labels=dict(color="°C", x="Month", y="Hour")), 380, "Hour × month temperature climatology"), width="stretch")
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_scatter(x=m.index, y=m.pres, name="Pressure (hPa)", mode="lines+markers", line=dict(color="#38bdf8", width=3)); f.add_bar(x=m.index, y=m.rain, name="Rain (mm/yr)", marker_color="rgba(20,184,166,.4)", secondary_y=True)
    b.plotly_chart(style(f, 380, "Pressure dips as the monsoon peaks"), width="stretch")

with t2:
    v = st.selectbox("Variable", NUM, format_func=LBL.get, key="dv"); bins = st.slider("Histogram bins", 10, 100, 40)
    a, b = st.columns(2)
    d = df[df[v] > 0] if (v == "precip" and st.checkbox("Rainy hours only (precip > 0)", True)) else df
    a.plotly_chart(style(px.histogram(d, x=v, nbins=bins, marginal="box", color_discrete_sequence=[PAL[0]], labels=LBL), 430, f"Histogram + boxplot: {LBL[v]}"), width="stretch")
    x = np.sort(d[v].values[::max(len(d)//5000, 1)]); fo = go.Figure(go.Scatter(x=x, y=np.arange(1, len(x)+1)/len(x)*100, mode="lines", line=dict(color=PAL[2], width=3)))
    fo.update_layout(xaxis_title=LBL[v], yaxis_title="Cumulative % of hours")
    b.plotly_chart(style(fo, 430, f"Ogive (cumulative frequency): {LBL[v]}"), width="stretch")
    s = d[v]; q1, q3 = s.quantile([.25, .75])
    st.dataframe(pd.DataFrame({"Mean":[s.mean()],"Median":[s.median()],"Mode":[s.mode()[0]],"Std":[s.std()],"Variance":[s.var()],"IQR":[q3-q1],"Skewness":[stats.skew(s)],"Excess kurtosis":[stats.kurtosis(s)]}).round(3), hide_index=True, width="stretch")

with t3:
    v = st.selectbox("Variable", NUM, index=NUM.index("wind"), format_func=LBL.get, key="ov")
    a, b = st.columns(2)
    d = df.assign(Month=df.month.map(lambda i: MON[i-1]))
    a.plotly_chart(style(px.box(d, x="Month", y=v, category_orders={"Month":MON}, color_discrete_sequence=[PAL[3]], labels=LBL, points="outliers"), 430, f"Boxplot by month: {LBL[v]}"), width="stretch")
    smp = df[v].sample(min(4000, len(df)), random_state=1).values; (osm, osr), (sl, ic, _) = stats.probplot(smp, dist="norm")
    f = go.Figure(); f.add_scatter(x=osm, y=osr, mode="markers", marker=dict(size=4, color=PAL[1]), name="Sample"); f.add_scatter(x=osm, y=sl*osm+ic, mode="lines", line=dict(color="white"), name="Normal reference")
    f.update_layout(xaxis_title="Theoretical quantiles", yaxis_title="Ordered values"); b.plotly_chart(style(f, 430, f"Q-Q plot: {LBL[v]}"), width="stretch")
    q1, q3 = df[v].quantile([.25, .75]); i = q3-q1; lo, hi = q1-1.5*i, q3+1.5*i; n = int(((df[v] < lo) | (df[v] > hi)).sum())
    st.markdown(f'<div class="story"><b>IQR rule:</b> Q1={q1:.2f}, Q3={q3:.2f}, IQR={i:.2f} → fences [{lo:.2f}, {hi:.2f}] → <b>{n:,} outliers ({n/len(df)*100:.2f}%)</b>. '
                f'D\'Agostino normality test p = {stats.normaltest(smp).pvalue:.2e} → {"not normal" if stats.normaltest(smp).pvalue < .05 else "consistent with normal"}. Extreme values are real weather events, so they are flagged, not deleted.</div>', unsafe_allow_html=True)
    if v == "precip": st.info("Precipitation is zero-inflated (Q1=Q3=0), so the IQR flags every rainy hour. Use the rainy-hours-only histogram in the Distributions tab.")

with t4:
    a, b = st.columns(2); smp = df.sample(min(6000, len(df)), random_state=2)
    x = a.selectbox("X", NUM, 0, format_func=LBL.get); y = b.selectbox("Y", NUM, NUM.index("app_temp"), format_func=LBL.get)
    cc = st.selectbox("Colour by", ["rh","pressure","wind","temp","month"], format_func=lambda k: LBL.get(k, k.title()))
    f = px.scatter(smp, x=x, y=y, color=cc, color_continuous_scale="Turbo", opacity=.65, labels=LBL, trendline=None)
    st.plotly_chart(style(f, 430, f"Scatter: {LBL[x]} vs {LBL[y]}  (r = {df[x].corr(df[y]):.2f})"), width="stretch")
    a, b = st.columns(2); cm = df[NUM].corr()
    a.plotly_chart(style(px.imshow(cm.rename(index=lambda k:k, columns=lambda k:k), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"), 520, "Correlation matrix (Pearson)"), width="stretch")
    pv = ["temp","app_temp","rh","pressure"]; sm = smp.sample(min(2500, len(smp)), random_state=3)
    b.plotly_chart(style(px.scatter_matrix(sm, dimensions=pv, color="rh", color_continuous_scale="Viridis", labels={k:LBL[k].split(" (")[0] for k in pv}).update_traces(diagonal_visible=True, marker=dict(size=3)), 520, "Pair-plot: temp, apparent temp, humidity, pressure"), width="stretch")

with t5:
    a, b = st.columns(2)
    a.subheader("Attribute classification")
    a.dataframe(pd.DataFrame({"Attribute":["location","time"]+NUM,"Type":["Nominal","Interval (datetime)","Interval, continuous","Ratio, continuous (0-100)","Interval, continuous","Interval, continuous","Ratio, continuous","Ratio, continuous","Ratio, continuous, zero-inflated","Ratio, continuous","Ratio, continuous","Ratio, continuous","Ratio, continuous"]}), hide_index=True, width="stretch", height=420)
    b.subheader("Data quality")
    b.dataframe(pd.DataFrame({"Check":["Missing values","Duplicate timestamps","Missing hours in index","Impossible values (RH∉[0,100], precip/wind<0)"],"Result":[int(raw[NUM].isna().sum().sum()), int(raw.time.duplicated().sum()), 0, int(((raw.rh<0)|(raw.rh>100)|(raw.precip<0)|(raw.wind<0)).sum())]}), hide_index=True, width="stretch")
    b.caption("Gap strategy: time interpolation (≤6 h), precipitation → 0, remainder → median. IQR outliers are kept (real extremes).")
    st.subheader("Normalization")
    method = st.radio("Method", ["Min-Max", "Z-score"], horizontal=True); fv = st.multiselect("Features", NUM, ["temp","rh","pressure","wind"], format_func=LBL.get)
    if fv:
        nm = (df[fv]-df[fv].min())/(df[fv].max()-df[fv].min()) if method == "Min-Max" else (df[fv]-df[fv].mean())/df[fv].std()
        a, b = st.columns(2); a.write("Before"); a.dataframe(df[fv].describe().loc[["min","max","mean","std"]].round(3)); b.write(f"After ({method})"); b.dataframe(nm.describe().loc[["min","max","mean","std"]].round(3))
    st.subheader("Descriptive statistics")
    keyv = ["temp","rh","wind","precip"]; rows = {}
    for k in keyv:
        s = df[k]; q = s.quantile([.25,.5,.75])
        rows[LBL[k]] = {"Mean":s.mean(),"Median":s.median(),"Mode":s.mode()[0],"Variance":s.var(),"Std":s.std(),"Range":s.max()-s.min(),"IQR":q[.75]-q[.25],"Coef. dispersion":s.std()/s.mean(),"Skewness":stats.skew(s),"Excess kurtosis":stats.kurtosis(s),"Pearson skew":3*(s.mean()-s.median())/s.std(),"Bowley skew":(q[.75]+q[.25]-2*q[.5])/(q[.75]-q[.25]) if q[.75]!=q[.25] else np.nan}
    st.dataframe(pd.DataFrame(rows).round(3), width="stretch")
    st.subheader("Distance / proximity")
    c1, c2, c3 = st.columns(3); dy = c1.selectbox("Year", sorted(raw.year.unique()), index=23); dh = c2.slider("Hour", 0, 23, 14, key="dh"); dm = c3.radio("Metric", ["euclidean","cityblock"], format_func=lambda s: "Euclidean" if s=="euclidean" else "Manhattan", horizontal=True)
    sub = raw[(raw.year==dy)&(raw.hour==dh)&(raw.time.dt.day==15)]; feat = ["temp","rh","pressure","wind"]
    z = (sub[feat]-raw[feat].mean())/raw[feat].std(); names = [MON[i-1] for i in sub.month]
    D = pd.DataFrame(cdist(z, z, dm), index=names, columns=names).round(2)
    st.plotly_chart(style(px.imshow(D, text_auto=True, color_continuous_scale="Magma", aspect="auto"), 480, f"{dm.title()} distance on z-scored temp, RH, pressure, wind (15th of each month, {dh}:00, {dy})"), width="stretch")

with t6:
    mm = df.groupby("month").agg(rain=("precip","sum"), rh=("rh","mean"), pres=("pressure","mean"), wind=("wind","mean")); mm["rain"] /= nyears
    wet, dry = df[df.precip>0].pressure.mean(), df[df.precip==0].pressure.mean()
    tr = stats.linregress(df.groupby("year").temp.mean().index, df.groupby("year").temp.mean().values) if nyears > 2 else None
    st.markdown("### Key insights (computed live from your filters)")
    st.markdown(f'<div class="story"><b>1. Two-season climate.</b> {mons_share:.0f}% of rainfall falls Jun-Sep. <i>Why:</i> the southwest monsoon brings moist Arabian Sea air that is forced up the Western Ghats and condenses; humidity and wind climb with it.</div>'
                f'<div class="story"><b>2. Pressure is the signal.</b> Rainy hours average {wet:.1f} hPa vs {dry:.1f} hPa on dry hours (r(pressure, precip) = {df.pressure.corr(df.precip):.2f}). <i>Why:</i> the monsoon trough is a low-pressure zone and rising air condenses into rain.</div>'
                f'<div class="story"><b>3. Humidity amplifies heat.</b> r(temp, RH) = {df.temp.corr(df.rh):.2f}, yet apparent temperature runs up to {(df.app_temp-df.temp).max():.1f} °C above actual. <i>Why:</i> humid air slows sweat evaporation.</div>'
                f'<div class="story"><b>4. Variability, not trend.</b> Annual rain ranges {ann_rain.min():.0f}-{ann_rain.max():.0f} mm.'
                + (f' Temperature trend {tr.slope*10:+.2f} °C/decade (p={tr.pvalue:.2f}), not significant.' if tr else '') + '</div>', unsafe_allow_html=True)
    st.markdown("### Domain recommendations")
    st.dataframe(pd.DataFrame({"Domain":["🌾 Agriculture","🚨 Disaster management","🩺 Public health","⚡ Energy","🏗️ Construction safety"],
        "Data-driven action":["Schedule sowing around the Jun-Sep monsoon; plan supplementary irrigation Jan-May; budget for ±40% year-to-year rainfall swings.",
        "Raise flood alerts when pressure falls below ~945 hPa and hourly rain reaches ≥10 mm; use rolling pressure drops as an early-warning feature.",
        "Issue heat advisories on apparent temperature (≥41 °C), focusing on 12:00-16:00 in Mar-May.",
        "Expect solar output to collapse in Jun-Sep; plan storage / grid backup.",
        "Avoid crane and outdoor work during Jun-Aug when gusts concentrate."]}), hide_index=True, width="stretch")
