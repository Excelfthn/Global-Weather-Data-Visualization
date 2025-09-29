import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Global Weather EDA", page_icon="🌍", layout="wide")

# ---------- Data loading ----------
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    # Casting datetime columns (kalau ada)
    dt_cols = [c for c in ["last_updated","sunrise","sunset"] if c in df.columns]
    for c in dt_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

# Kalau file ada di repo GitHub, cukup pakai nama file
DATA_PATH = "Global_Weather.csv"
df = load_data(DATA_PATH)

st.title("🌍 Global Weather EDA Dashboard")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
countries = sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns else []
sel_countries = st.sidebar.multiselect("Country", countries, default=countries[:10] if countries else [])
hour_min, hour_max = st.sidebar.slider("Hour of Day (local)", 0, 23, (0,23))

# Normalisasi label negara (opsional, contoh kecil)
country_mapping = {
    "Saudi Arabien": "Saudi Arabia",
    "Marrocos": "Morocco",
    "Турция": "Turkey",
    "Inde": "India",
}
if "country" in df.columns:
    df["country"] = df["country"].replace(country_mapping)

# Subset berdasarkan filter
d = df.copy()
if sel_countries:
    d = d[d["country"].isin(sel_countries)]
if "last_updated" in d.columns:
    d["hour"] = d["last_updated"].dt.hour
    d = d[(d["hour"] >= hour_min) & (d["hour"] <= hour_max)]

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Map (Scatter Mapbox)",
    "Precip Histogram (Log-Y)",
    "Temp vs Humidity (Density)",
    "Temp by Hour (Mean ± 95% CI)",
    "Mean Temp by Country",
    "Day vs Night (Pie)"
])

# ---------- 1) Map ----------
with tab1:
    st.subheader("Global Temperature – Scatter Map (zoom/pan)")
    if {"latitude","longitude","temperature_celsius"}.issubset(d.columns):
        # clamp color to 1–99th percentile to avoid outlier-wash
        tmin, tmax = np.nanpercentile(d["temperature_celsius"].dropna(), [1, 99]) if d["temperature_celsius"].notna().any() else (0,1)
        fig = px.scatter_mapbox(
            d.dropna(subset=["latitude","longitude","temperature_celsius"]),
            lat="latitude", lon="longitude",
            color="temperature_celsius",
            color_continuous_scale="RdBu_r",
            range_color=(tmin, tmax),
            hover_name="location_name" if "location_name" in d.columns else None,
            hover_data={"country": True, "humidity": ":.0f", "uv_index":":.1f"} if "humidity" in d.columns and "uv_index" in d.columns else None,
            height=600, zoom=1.2
        )
        fig.update_layout(mapbox_style="open-street-map",
                          margin=dict(l=0,r=0,t=0,b=0),
                          coloraxis_colorbar=dict(title="Temp (°C)"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Columns needed: latitude, longitude, temperature_celsius.")

# ---------- 2) Histogram Precip (log-y) ----------
with tab2:
    st.subheader("Precipitation (mm) — Log-scaled Y")
    if "precip_mm" in d.columns:
        precip_pos = d.loc[d["precip_mm"].notna() & (d["precip_mm"] > 0), "precip_mm"]
        zero_count = int((d["precip_mm"] == 0).sum()) if "precip_mm" in d.columns else 0
        fig = px.histogram(precip_pos, x="precip_mm", nbins=50, template="plotly_white", opacity=0.9)
        fig.update_traces(marker_line_color="rgba(0,0,0,0.5)", marker_line_width=0.3,
                          hovertemplate="<b>Bin</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>")
        fig.update_layout(yaxis_type="log", xaxis_title="Precipitation (mm)", yaxis_title="Count (log)")
        if zero_count > 0:
            fig.add_annotation(text=f"{zero_count:,} zero-mm observations not shown on log scale",
                               xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False, font=dict(size=11, color="gray"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Column needed: precip_mm.")

# ---------- 3) Temp vs Humidity (density/hexbin-like) ----------
with tab3:
    st.subheader("Temperature vs Humidity — Density")
    if {"temperature_celsius","humidity"}.issubset(d.columns):
        # gunakan density_contour untuk alternatif hexbin
        fig = px.density_heatmap(
            d.dropna(subset=["temperature_celsius","humidity"]),
            x="temperature_celsius", y="humidity",
            nbinsx=40, nbinsy=40, color_continuous_scale="Viridis", height=520
        )
        fig.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Humidity (%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Columns needed: temperature_celsius, humidity.")

# ---------- 4) Temp by Hour (Mean ± 95% CI) ----------
with tab4:
    st.subheader("Average Temperature by Hour — Mean ± 95% CI")
    if {"last_updated","temperature_celsius"}.issubset(d.columns):
        dt = d.dropna(subset=["last_updated","temperature_celsius"]).copy()
        dt["hour"] = dt["last_updated"].dt.hour
        g = dt.groupby("hour")["temperature_celsius"].agg(["mean","std","count"]).reset_index()
        g["se"] = g["std"] / np.sqrt(g["count"].replace(0, np.nan))
        g["ci95"] = 1.96 * g["se"]
        g["lo"] = g["mean"] - g["ci95"]; g["hi"] = g["mean"] + g["ci95"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["hour"], y=g["lo"], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=g["hour"], y=g["hi"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(68,74,150,0.15)",
                                 name="95% CI"))
        fig.add_trace(go.Scatter(x=g["hour"], y=g["mean"], mode="lines+markers",
                                 line=dict(width=2, color="#444a96"),
                                 marker=dict(size=6, color="#444a96"),
                                 name="Mean"))
        fig.update_layout(xaxis_title="Hour (0–23, local)", yaxis_title="Temperature (°C)",
                          template="plotly_white", height=520)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Columns needed: last_updated, temperature_celsius.")

# ---------- 5) Mean Temp by Country (Top-15, colored tiers) ----------
with tab5:
    st.subheader("Top-15 Countries by Mean Temperature (°C)")
    if {"country","temperature_celsius"}.issubset(d.columns):
        mean_series = d.groupby("country")["temperature_celsius"].mean().sort_values(ascending=False).head(15)
        # warna bertingkat
        def lighten(hex_color, factor=0.4):
            hex_color = hex_color.lstrip("#")
            r, g, b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
            r = int(r + (255-r)*factor); g = int(g + (255-g)*factor); b = int(b + (255-b)*factor)
            return f"#{r:02x}{g:02x}{b:02x}"
        base = "#444a96"; light = lighten(base, 0.35); lighter = lighten(base, 0.65)
        colors_desc = [base] + [light]*4 + [lighter]*10
        top15_plot = mean_series.iloc[::-1]
        colors_plot = colors_desc[::-1]

        fig = go.Figure(go.Bar(
            x=top15_plot.values, y=top15_plot.index.astype(str),
            orientation="h",
            marker=dict(color=colors_plot, line=dict(color="rgba(0,0,0,0.3)", width=0.5)),
            text=[f"{v:.1f}°C" for v in top15_plot.values], textposition="outside",
            hovertemplate="<b>%{y}</b><br>Mean Temp: %{x:.2f} °C<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Mean Temperature (°C)", yaxis_title="", template="plotly_white", height=520)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Columns needed: country, temperature_celsius.")

# ---------- 6) Day vs Night (Pie) ----------
with tab6:
    st.subheader("Share of Observations: Day vs Night")
    if {"last_updated","sunrise","sunset"}.issubset(d.columns):
        dt = d.dropna(subset=["last_updated","sunrise","sunset"]).copy()
        def classify_day_night(row):
            su, ss, lu = row["sunrise"], row["sunset"], row["last_updated"]
            if pd.notna(su) and pd.notna(ss) and (su < ss):
                return "Day" if (lu.time() >= su.time()) and (lu.time() < ss.time()) else "Night"
            return np.nan
        dt["day_night"] = dt.apply(classify_day_night, axis=1)
        counts = dt["day_night"].value_counts(dropna=True)
        fig = px.pie(names=counts.index, values=counts.values, hole=0.2, color=counts.index,
                     color_discrete_map={"Day":"#f39c12","Night":"#2c3e50"})
        fig.update_traces(textposition='inside', texttemplate="%{label}<br>%{percent:.1%}")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Columns needed: last_updated, sunrise, sunset.")
