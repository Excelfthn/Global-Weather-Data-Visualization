import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Global Weather — One Page", page_icon="🌍", layout="wide")
PRIMARY = "#444a96"

# Load 
@st.cache_data
def load_data(src):
    df = pd.read_csv(src)
    for c in ["last_updated","sunrise","sunset"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    mapping = {"Saudi Arabien":"Saudi Arabia","Marrocos":"Morocco","Турция":"Turkey","Inde":"India"}
    if "country" in df.columns:
        df["country"] = df["country"].replace(mapping)
    return df

DATA_PATH = "Global_Weather.csv"
df = load_data(DATA_PATH)

# Sidebar 
st.sidebar.header("Country Filter")
if "country" in df.columns:
    all_countries = sorted(df["country"].dropna().unique().tolist())
    sel_countries = st.sidebar.multiselect("Select countries", all_countries, default=all_countries)
else:
    sel_countries = []

st.sidebar.markdown("---")
st.sidebar.header("Show / Hide Visualizations")

# Default True: all visible
show_raw              = st.sidebar.checkbox("Raw Data Preview", True)
show_dtypes_missing   = st.sidebar.checkbox("Data Types & Missing Values", True)
show_numeric_summary  = st.sidebar.checkbox("Numeric Summary", True)
show_hist_precip      = st.sidebar.checkbox("Precipitation (mm) Distribution (Log-scaled Y)", True)
show_hex_temp_hum     = st.sidebar.checkbox("Temperature vs Humidity (Hexbin Density)", True)
show_corr_heatmap     = st.sidebar.checkbox("Correlation Heatmap — Main Weather Variables", True)
show_line_hour_ci     = st.sidebar.checkbox("Average Temperature by Hour (Mean ± 95% CI)", True)
show_bar_median_temp  = st.sidebar.checkbox("Top-15 Countries by Median Temperature (°C)", True)
show_bar_mean_temp    = st.sidebar.checkbox("Top-15 Countries by Mean Temperature (°C)", True)
show_bar_median_hum   = st.sidebar.checkbox("Top-15 Countries by Median Humidity (%)", True)
show_scatter_latlon   = st.sidebar.checkbox("Global Temperature Map (Lat-Lon)", True)
show_scatter_mapbox   = st.sidebar.checkbox("Global Temperature – Scatter Map (Zoom & Pan)", True)
show_box_daynight     = st.sidebar.checkbox("Temperature by Day vs Night (Boxplot)", True)
show_pie_daynight     = st.sidebar.checkbox("Share of Observations: Day vs Night (Pie)", True)
show_wind_rose        = st.sidebar.checkbox("Wind Rose (avg wind_kph by direction sector)", True)

# Apply country filter 
d = df.copy()
if sel_countries:
    d = d[d["country"].isin(sel_countries)]

# Page
st.title("🌍 Global Weather — One Page EDA & Visualizations")

# Raw Data Preview 
if show_raw:
    st.subheader("Raw Data Preview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Rows", f"{len(d):,}")
    with c2: st.metric("Columns", d.shape[1])
    with c3: st.metric("Countries", int(d["country"].nunique()) if "country" in d.columns else 0)
    with c4: st.metric("Locations", int(d["location_name"].nunique()) if "location_name" in d.columns else 0)
    st.dataframe(d.head(25), use_container_width=True)

# Data Types & Missing 
if show_dtypes_missing:
    st.subheader("Data Types & Missing Values")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**dtypes**")
        st.dataframe(pd.DataFrame(d.dtypes, columns=["dtype"]))
    with colB:
        st.markdown("**Missing by Column**")
        mv = d.isna().sum().to_frame("missing")
        mv["missing_%"] = (mv["missing"]/len(d)*100).round(2)
        st.dataframe(mv.sort_values("missing_%", ascending=False))

# Numeric Summary 
if show_numeric_summary:
    st.subheader("Numeric Summary")
    num_cols = d.select_dtypes(include="number").columns.tolist()
    if num_cols:
        st.dataframe(d[num_cols].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns available.")

st.markdown("---")

# Precipitation Histogram (log-Y) 
if show_hist_precip:
    st.subheader("Precipitation (mm) Distribution (Log-scaled Y)")
    if "precip_mm" in d.columns:
        zeros = int((d["precip_mm"] == 0).sum())
        pos = d.loc[(d["precip_mm"] > 0) & d["precip_mm"].notna(), "precip_mm"]
        if len(pos) > 0:
            fig = px.histogram(pos, x="precip_mm", nbins=50, template="plotly_white")
            fig.update_traces(marker_color=PRIMARY, marker_line_color="rgba(0,0,0,0.45)", marker_line_width=0.4)
            fig.update_layout(yaxis_type="log", xaxis_title="Precipitation (mm)", yaxis_title="Count (log)")
            if zeros > 0:
                fig.add_annotation(text=f"{zeros:,} zero-mm rows omitted on log scale",
                                   xref="paper", yref="paper", x=0.5, y=1.06, showarrow=False,
                                   font=dict(size=11, color="gray"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positive precipitation values to plot.")
    else:
        st.info("Column 'precip_mm' is missing.")

# Temperature vs Humidity (Hexbin/Density) 
if show_hex_temp_hum:
    st.subheader("Temperature vs Humidity (Hexbin Density)")
    need = {"temperature_celsius","humidity"}
    if need.issubset(d.columns):
        dd = d.dropna(subset=list(need))
        if not dd.empty:
            fig = px.density_heatmap(dd, x="temperature_celsius", y="humidity",
                                     nbinsx=40, nbinsy=40, color_continuous_scale="Viridis", height=520)
            fig.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Humidity (%)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rows available after filtering.")
    else:
        st.info("Need columns: temperature_celsius, humidity.")

# Correlation Heatmap 
if show_corr_heatmap:
    st.subheader("Correlation Heatmap — Main Weather Variables")
    cols = [c for c in [
        "temperature_celsius","humidity","pressure_mb","wind_kph","gust_kph",
        "precip_mm","cloud","visibility_km","uv_index","moon_illumination"
    ] if c in d.columns]
    if len(cols) >= 2:
        corr = d[cols].corr(method="pearson")
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                        labels=dict(color="Pearson r"))
        fig.update_layout(template="plotly_white", height=520)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation.")

# Average Temperature by Hour (Mean ± 95% CI) 
if show_line_hour_ci:
    st.subheader("Average Temperature by Hour (Mean ± 95% CI)")
    need = {"last_updated","temperature_celsius"}
    if need.issubset(d.columns):
        dt = d.dropna(subset=list(need)).copy()
        dt["hour"] = dt["last_updated"].dt.hour
        g = dt.groupby("hour")["temperature_celsius"].agg(["mean","std","count"]).reset_index()
        g["se"] = g["std"] / np.sqrt(g["count"].replace(0, np.nan))
        g["ci95"] = 1.96 * g["se"]; g["lo"] = g["mean"] - g["ci95"]; g["hi"] = g["mean"] + g["ci95"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["hour"], y=g["lo"], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=g["hour"], y=g["hi"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(68,74,150,0.18)", name="95% CI"))
        fig.add_trace(go.Scatter(x=g["hour"], y=g["mean"], mode="lines+markers",
                                 line=dict(color=PRIMARY, width=2), marker=dict(size=6, color=PRIMARY),
                                 name="Mean"))
        fig.update_layout(template="plotly_white", xaxis_title="Hour (0–23, local)", yaxis_title="Temperature (°C)", height=520)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need columns: last_updated, temperature_celsius.")

st.markdown("---")

# Top-15 Countries by Median Temperature 
if show_bar_median_temp:
    st.subheader("Top-15 Countries by Median Temperature (°C)")
    need = {"country","temperature_celsius"}
    if need.issubset(d.columns):
        s = d.groupby("country")["temperature_celsius"].median().sort_values(ascending=False).head(15)
        top = s.iloc[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index.astype(str), orientation="h",
                               marker=dict(color="#888ad6"),
                               text=[f"{v:.1f}°C" for v in top.values], textposition="outside"))
        fig.update_layout(template="plotly_white", xaxis_title="Median Temperature (°C)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need columns: country, temperature_celsius.")

# Top-15 Countries by Mean Temperature (tiered colors) 
if show_bar_mean_temp:
    st.subheader("Top-15 Countries by Mean Temperature (°C)")
    need = {"country","temperature_celsius"}
    if need.issubset(d.columns):
        mean_series = d.groupby("country")["temperature_celsius"].mean().sort_values(ascending=False).head(15)
        def lighten(hex_color, factor=0.35):
            h = hex_color.lstrip("#"); r,g,b = int(h[:2],16), int(h[2:4],16), int(h[4:],16)
            r = int(r + (255-r)*factor); g = int(g + (255-g)*factor); b = int(b + (255-b)*factor)
            return f"#{r:02x}{g:02x}{b:02x}"
        base, light, lighter = PRIMARY, lighten(PRIMARY,0.35), lighten(PRIMARY,0.65)
        colors_desc = [base] + [light]*4 + [lighter]*10
        top = mean_series.iloc[::-1]; colors_plot = colors_desc[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index.astype(str), orientation="h",
                               marker=dict(color=colors_plot, line=dict(color="rgba(0,0,0,0.3)", width=0.4)),
                               text=[f"{v:.1f}°C" for v in top.values], textposition="outside"))
        fig.update_layout(template="plotly_white", xaxis_title="Mean Temperature (°C)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need columns: country, temperature_celsius.")

# Top-15 Countries by Median Humidity 
if show_bar_median_hum:
    st.subheader("Top-15 Countries by Median Humidity (%)")
    need = {"country","humidity"}
    if need.issubset(d.columns):
        s = d.groupby("country")["humidity"].median().sort_values(ascending=False).head(15)
        top = s.iloc[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index.astype(str), orientation="h",
                               marker=dict(color=PRIMARY),
                               text=[f"{v:.0f}%" for v in top.values], textposition="outside"))
        fig.update_layout(template="plotly_white", xaxis_title="Median Humidity (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need columns: country, humidity.")

# Global Temperature Map (Lat–Lon) 
if show_scatter_latlon:
    st.subheader("Global Temperature Map (Lat-Lon)")
    need = {"latitude","longitude","temperature_celsius"}
    if need.issubset(d.columns):
        dd = d.dropna(subset=list(need))
        fig = px.scatter(dd, x="longitude", y="latitude",
                         color="temperature_celsius", color_continuous_scale="RdBu_r", height=520)
        fig.update_layout(template="plotly_white", xaxis_title="Longitude", yaxis_title="Latitude",
                          coloraxis_colorbar=dict(title="Temp (°C)"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need columns: latitude, longitude, temperature_celsius.")

# Global Temperature — Scatter Mapbox (Zoom & Pan) 
if show_scatter_mapbox:
    st.subheader("Global Temperature – Scatter Map (Zoom & Pan)")
    need = {"latitude","longitude","temperature_celsius"}
    if need.issubset(d.columns):
        dm = d.dropna(subset=list(need)).copy()
        if not dm.empty:
            tmin, tmax = np.nanpercentile(dm["temperature_celsius"], [1, 99])
            fig = px.scatter_mapbox(dm, lat="latitude", lon="longitude",
                                    color="temperature_celsius", color_continuous_scale="RdBu_r",
                                    range_color=(tmin, tmax), zoom=1.2, height=520,
                                    hover_name="location_name" if "location_name" in dm.columns else None,
                                    hover_data={"country": True} if "country" in dm.columns else None)
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0),
                              coloraxis_colorbar=dict(title="Temp (°C)"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid geospatial rows after filtering.")
    else:
        st.info("Need columns: latitude, longitude, temperature_celsius.")

# Temperature by Day vs Night (Boxplot) 
if show_box_daynight:
    st.subheader("Temperature by Day vs Night (Boxplot)")
    need = {"last_updated","sunrise","sunset","temperature_celsius"}
    if need.issubset(d.columns):
        dt = d.dropna(subset=list(need)).copy()
        def dn(row):
            su, ss, lu = row["sunrise"], row["sunset"], row["last_updated"]
            if pd.notna(su) and pd.notna(ss) and (su < ss):
                return "Day" if (lu.time() >= su.time()) and (lu.time() < ss.time()) else "Night"
            return np.nan
        dt["day_night"] = dt.apply(dn, axis=1)
        dt = dt.dropna(subset=["day_night"])
        if not dt.empty:
            fig = px.box(dt, x="day_night", y="temperature_celsius", color="day_night",
                         color_discrete_map={"Day":PRIMARY,"Night":"#9999cc"}, template="plotly_white")
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Temperature (°C)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rows labeled Day/Night after processing.")
    else:
        st.info("Need columns: last_updated, sunrise, sunset, temperature_celsius.")

# Share of Observations: Day vs Night (Pie) 
if show_pie_daynight:
    st.subheader("Share of Observations: Day vs Night")
    need = {"last_updated","sunrise","sunset"}
    if need.issubset(d.columns):
        dt = d.dropna(subset=list(need)).copy()
        def dn(row):
            su, ss, lu = row["sunrise"], row["sunset"], row["last_updated"]
            if pd.notna(su) and pd.notna(ss) and (su < ss):
                return "Day" if (lu.time() >= su.time()) and (lu.time() < ss.time()) else "Night"
            return np.nan
        dt["day_night"] = dt.apply(dn, axis=1)
        vc = dt["day_night"].value_counts(dropna=True)
        if len(vc) > 0:
            fig = px.pie(names=vc.index, values=vc.values, hole=0.25,
                         color=vc.index, color_discrete_map={"Day":"#f39c12","Night":"#2c3e50"})
            fig.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent:.1%}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Day/Night counts to plot.")
    else:
        st.info("Need columns: last_updated, sunrise, sunset.")

# Wind Rose (avg wind_kph)
if show_wind_rose:
    st.subheader("Wind Rose (avg wind_kph by direction sector)")
    need = {"wind_degree","wind_kph"}
    if need.issubset(d.columns):
        dw = d.dropna(subset=list(need)).copy()
        if not dw.empty:
            sectors = np.arange(0, 360, 22.5)
            dw["sector"] = pd.cut(dw["wind_degree"] % 360, bins=np.append(sectors, 360),
                                  right=False, include_lowest=True)
            g = dw.groupby("sector")["wind_kph"].mean().reset_index()
            g["theta"] = g["sector"].apply(lambda x: (x.left + x.right)/2 if pd.notna(x) else np.nan)
            fig = go.Figure(go.Barpolar(theta=g["theta"], r=g["wind_kph"],
                                        marker=dict(color=PRIMARY, line=dict(color="white", width=1))))
            fig.update_layout(template="plotly_white", height=520,
                              polar=dict(radialaxis=dict(angle=0, tickangle=0)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No wind data to plot.")
    else:
        st.info("Need columns: wind_degree, wind_kph.")

st.markdown("---")
st.caption("Order: raw → quality → summaries → distributions → relationships → temporal/spatial → categorical → wind.")

