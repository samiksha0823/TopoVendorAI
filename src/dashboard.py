import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIG & OBSIDIAN DARK THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="TopoVendor AI — SOC Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling and Anchor Links
st.markdown("""
<style>
    /* Dark Deep Space Palette */
    :root {
        --bg-deep: #070B1A;
        --bg-panel: #0E1530;
        --bg-card: #131B3D;
        --teal: #2DD9C4;
        --green: #3ECF8E;
        --amber: #fad046;
        --red: #d42c26;
        --text-primary: #EAF0FB;
        --text-muted: #8A93B8;
        --border-subtle: rgba(148, 163, 209, 0.14);
    }

    body, .stApp {
        background-color: var(--bg-deep) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif;
    }

    /* Metric Cards */
    .metric-card {
        background-color: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-muted);
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-primary);
    }
    .metric-value.critical { color: var(--red); }
    .metric-value.active { color: var(--teal); }

    /* Custom Container Panel */
    .panel-box {
        background-color: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Alert Feed Pill */
    .alert-pill-critical {
        background-color: rgba(240, 87, 107, 0.15);
        border-left: 4px solid var(--red);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 4px;
    }
    .alert-pill-elevated {
        background-color: rgba(242, 184, 75, 0.15);
        border-left: 4px solid var(--amber);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 4px;
    }

    /* Reason Pills */
    .reason-badge {
        display: inline-block;
        background: rgba(45, 217, 196, 0.12);
        color: var(--teal);
        border: 1px solid rgba(45, 217, 196, 0.3);
        border-radius: 16px;
        padding: 6px 14px;
        font-size: 12px;
        margin: 4px 6px 4px 0;
    }
    
    /* Sidebar Navigation Links */
    .sidebar-nav-link {
        display: block;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-radius: 8px;
        color: #EAF0FB !important;
        background-color: #0E1530;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.2s;
        border: 1px solid rgba(148, 163, 209, 0.14);
    }
    .sidebar-nav-link:hover, .sidebar-nav-link:active {
        background-color: #131B3D;
        border-color: #2DD9C4;
        color: #2DD9C4 !important;
        box-shadow: 0 0 10px rgba(45,217,196,0.15);
    }

    /* Hide default Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Anchor offset for fixed header */
    h2 {
        scroll-margin-top: 80px; 
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA PIPELINE LOADER
# ==============================================================================
@st.cache_data
def load_pipeline_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inc_path = os.path.join(project_root, "output", "vendor_incident_response.csv")
    risk_path = os.path.join(project_root, "output", "vendor_dynamic_risk_scores.csv")

    if os.path.exists(inc_path) and os.path.exists(risk_path):
        df_inc = pd.read_csv(inc_path)
        df_risk = pd.read_csv(risk_path)
        df = df_inc if 'dynamic_risk_score' in df_inc.columns else pd.merge(df_risk, df_inc, on=["window_start", "vendor_id"])
    elif os.path.exists(risk_path):
        df = pd.read_csv(risk_path)
        df["recommended_actions"] = "No action specified"
        df["action_priority"] = np.where(df["risk_tier"] == "High", "Critical", "Routine")
    else:
        timestamps = pd.date_range("2026-07-25", periods=168, freq="1h")
        vendors = [f"VEND-100{i}" for i in range(1, 9)]
        records = []
        for t in timestamps:
            for v in vendors:
                beh_score = np.random.uniform(0.05, 0.8)
                topo_score = np.random.uniform(0.05, 0.6)
                score = 1 - (1 - beh_score) * (1 - topo_score)
                tier = "High" if score >= 0.65 else ("Medium" if score >= 0.35 else "Low")
                priority = "Critical" if tier == "High" else ("Elevated" if tier == "Medium" else "Routine")
                
                attack_type = ""
                if score > 0.7: attack_type = np.random.choice(["SSH-Patator", "DoS Hulk", "DoS GoldenEye", "DDoS", "PortScan", "FTP-Patator"])
                
                records.append({
                    "window_start": str(t),
                    "vendor_id": v,
                    "behavioral_component": round(beh_score, 3),
                    "topological_component": round(topo_score, 3),
                    "dynamic_risk_score": round(score, 3),
                    "risk_tier": tier,
                    "risk_reasons": "Access from an unfamiliar location; Activity outside normal working hours",
                    "recommended_actions": "Flag for SOC analyst review; Increase monitoring frequency for this vendor",
                    "action_priority": priority,
                    "vendor_attack_types_this_window": attack_type
                })
        df = pd.DataFrame(records)

    df["window_start"] = pd.to_datetime(df["window_start"])
    return df

df = load_pipeline_data()

# Temporal snapshots
max_time = df["window_start"].max()
last_24h_time = max_time - pd.Timedelta(hours=24)
df_latest = df[df["window_start"] == max_time].copy()

# ==============================================================================
# 3. SIDEBAR PANEL 
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap: 12px; margin-bottom: 30px;">
        <div style="width: 36px; height: 36px; background: linear-gradient(135deg, #2DD9C4, #4A6CF7); border-radius: 8px; display:flex; align-items:center; justify-content:center; font-weight:bold; color:#070B1A;">TV</div>
        <h2 style="margin:0; font-size: 20px; color: #EAF0FB;">Topo Vendor AI</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <a href="#soc-overview" class="sidebar-nav-link" target="_self">SOC Overview</a>
    <a href="#alert-queue" class="sidebar-nav-link" target="_self">Alert Queue</a>
    <a href="#vendor-hub" class="sidebar-nav-link" target="_self">Vendor Hub</a>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    <div style="display:flex; align-items:center; gap: 12px;">
        <img src="https://ui-avatars.com/api/?name=M+T&background=2DD9C4&color=070B1A&bold=true" style="border-radius: 50%; width: 42px;">
        <div>
            <div style="font-weight: 700; font-size: 14px; color: #EAF0FB;">M. Tarunima Rao</div>
            <div style="font-size: 12px; color: #8A93B8;">Lead SOC Analyst</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. SECTION: SOC OVERVIEW 
# ==============================================================================
st.markdown('<div style="color: #8A93B8; font-size: 16px; margin-bottom: -5px; font-weight: 500;">TopoVendor AI - Intelligent third party vendor assessment and threat detection</div>', unsafe_allow_html=True)
st.markdown('<h2 id="soc-overview" style="margin-top: 0;">SOC Overview</h2>', unsafe_allow_html=True)

col_m1, col_m2, col_m3, col_m4 = st.columns([1, 1.1, 1.8, 1])

total_vendors = df["vendor_id"].nunique()
total_events = 2830743  
active_vendors = (df_latest["dynamic_risk_score"] > 0.05).sum()

crit_week = (df["risk_tier"] == "High").sum()
crit_24h = ((df["window_start"] >= last_24h_time) & (df["risk_tier"] == "High")).sum()
crit_live = (df_latest["risk_tier"] == "High").sum()

with col_m1:
    st.markdown(f"""
    <div class="metric-card"><div class="metric-title">Total Vendors</div><div class="metric-value">{total_vendors}</div></div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card"><div class="metric-title">Events Analyzed</div><div class="metric-value">{total_events:,}</div></div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Critical Exposure</div>
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top: 4px;">
            <div style="text-align:center;"><div style="font-size:10px; color:var(--text-muted); margin-bottom:2px;">LIVE</div><div class="metric-value critical" style="font-size:24px;">{crit_live}</div></div>
            <div style="text-align:center; border-left: 1px solid rgba(148,163,209,0.2); padding-left: 20px;"><div style="font-size:10px; color:var(--text-muted); margin-bottom:2px;">LAST 24H</div><div class="metric-value critical" style="font-size:24px;">{crit_24h}</div></div>
            <div style="text-align:center; border-left: 1px solid rgba(148,163,209,0.2); padding-left: 20px;"><div style="font-size:10px; color:var(--text-muted); margin-bottom:2px;">FULL WEEK</div><div class="metric-value critical" style="font-size:24px;">{crit_week}</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
    <div class="metric-card"><div class="metric-title">Active Vendors (Live)</div><div class="metric-value active">{active_vendors}</div></div>
    """, unsafe_allow_html=True)

col_mid_left, col_mid_right = st.columns([1.5, 1])

with col_mid_left:
    st.markdown("### Risk Distribution")
    tier_counts = df["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    
    fig_dist = go.Figure(data=[go.Bar(
        y=["CRITICAL / HIGH", "MEDIUM", "LOW"],
        x=[tier_counts["High"], tier_counts["Medium"], tier_counts["Low"]],  
        orientation='h',
        marker=dict(color=["#d42c26", "#fad046", "#3ECF8E"]),
        text=[int(tier_counts["High"]), int(tier_counts["Medium"]), int(tier_counts["Low"])],
        textposition='auto',
    )])
    fig_dist.update_layout(plot_bgcolor="#0E1530", paper_bgcolor="#0E1530", font=dict(color="#EAF0FB"), margin=dict(l=20, r=20, t=10, b=20), height=280)
    st.plotly_chart(fig_dist, use_container_width=True)

with col_mid_right:
    st.markdown("### Recent Threats")
    if "vendor_attack_types_this_window" in df.columns:
        threats = df['vendor_attack_types_this_window'].dropna().astype(str)
        threats = threats.str.split(';').explode().str.strip()
        threats = threats[(threats != '') & (threats != 'BENIGN') & (threats != 'nan')]
        threat_counts = threats.value_counts()
    else:
        threat_counts = pd.Series()
    
    # List of vibrant colors for the threat dots
    dot_colors = ["#6C5DD3", "#3ECF8E", "#F2B84B", "#F0576B", "#2DD9C4", "#E83A82", "#00C3F8", "#FF9F43"]
    
    threat_html = '<div class="panel-box" style="height: 280px; overflow-y: auto;">\n'
    if threat_counts.empty:
        threat_html += "<div style='color:#8A93B8; margin-top:10px;'>No attacks registered in current dataset.</div>\n"
    else:
        for i, (threat, count) in enumerate(threat_counts.items()):
            color = dot_colors[i % len(dot_colors)]
            threat_html += f'<div style="display:flex; justify-content:space-between; padding: 12px 0; border-bottom: 1px solid rgba(148,163,209,0.1);">\n'
            threat_html += f'<span style="font-size: 14px;"><span style="color:{color}; margin-right: 12px; font-size: 14px;">●</span> {threat}</span>\n'
            threat_html += f'<span style="font-weight:bold; font-family: monospace; font-size: 15px;">{count}</span>\n'
            threat_html += f'</div>\n'
    threat_html += '</div>'
    st.markdown(threat_html, unsafe_allow_html=True)


st.markdown("---")

# ==============================================================================
# 5. SECTION: ALERT QUEUE
# ==============================================================================
st.markdown('<h2 id="alert-queue">Alert Queue</h2>', unsafe_allow_html=True)

col_alert_l, col_alert_r = st.columns([1.3, 1])

alerts_df = df[df["action_priority"].isin(["Critical", "Elevated"])].sort_values("window_start", ascending=False)

with col_alert_l:
    st.markdown("### Live Alert Field")
    if alerts_df.empty:
        st.info("No critical or elevated alerts registered this week.")
    else:
        with st.container(height=500):
            for idx, row in alerts_df.iterrows():
                col_a1, col_a2 = st.columns([3.5, 1.2])
                css_class = "alert-pill-critical" if row["action_priority"] == "Critical" else "alert-pill-elevated"
                pill_color = "#d42c26" if row["action_priority"] == "Critical" else "#fad046"
                
                with col_a1:
                    st.markdown(f"""
                    <div class="{css_class}">
                        <div style="display:flex; justify-content:space-between; font-weight:bold;">
                            <span>{row['vendor_id']}</span>
                            <span style="color:{pill_color}; font-size:11px; text-transform:uppercase;">{row['action_priority']} ({row['dynamic_risk_score']:.2f})</span>
                        </div>
                        <div style="font-size: 11px; color: #8A93B8; margin-top: 4px;">{str(row['window_start'])[5:16]} · {row['risk_reasons']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_a2:
                    is_checked = st.checkbox("Review", key=f"alert_chk_{idx}")
                    if is_checked:
                        st.markdown("<div style='color:#3ECF8E; font-weight:600; font-size:14px; margin-top: -10px;'>✅ Analyzed</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='color:#8A93B8; font-weight:600; font-size:14px; margin-top: -10px;'>🕒 Pending</div>", unsafe_allow_html=True)
                st.markdown("<hr style='margin: 4px 0 12px 0; border-color: rgba(148,163,209,0.1);'>", unsafe_allow_html=True)

with col_alert_r:
    st.markdown("### Alert Groups")
    pending_crit = 0
    pending_elev = 0
    reviewed_count = 0
    
    for idx, row in alerts_df.iterrows():
        if st.session_state.get(f"alert_chk_{idx}", False):
            reviewed_count += 1
        else:
            if row["action_priority"] == "Critical":
                pending_crit += 1
            else:
                pending_elev += 1
                
    if pending_crit == 0 and pending_elev == 0 and reviewed_count == 0:
        labels, values, colors = ["Clear"], [1], ["#3ECF8E"]
    else:
        labels = ['Pending Critical', 'Pending Elevated', 'Reviewed']
        values = [pending_crit, pending_elev, reviewed_count]
        colors = ['#d42c26', '#fad046', '#3ECF8E']

    fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6, marker=dict(colors=colors))])
    fig_donut.update_layout(plot_bgcolor="#0E1530", paper_bgcolor="#0E1530", font=dict(color="#EAF0FB"), margin=dict(l=20, r=20, t=10, b=20), height=340, showlegend=True)
    st.plotly_chart(fig_donut, use_container_width=True)

st.markdown("---")

# ==============================================================================
# 6. SECTION: VENDOR HUB 
# ==============================================================================
st.markdown('<h2 id="vendor-hub">Vendor Hub</h2>', unsafe_allow_html=True)

selected_vendor = st.selectbox("Select Vendor ID to Investigate:", sorted(df["vendor_id"].unique()))

v_df = df[df["vendor_id"] == selected_vendor].sort_values("window_start")
v_latest = v_df.iloc[-1]

# --- TOP ROW: GRAPHS (Side-by-side) ---
col_g1, col_g2 = st.columns([1.3, 1])

with col_g1:
    st.markdown(f"#### Risk Trendline: `{selected_vendor}`")
    
    fig_trend = go.Figure()
    
    if "behavioral_component" in v_df.columns:
        fig_trend.add_trace(go.Scatter(
            x=v_df["window_start"], y=v_df["behavioral_component"],
            name="Behavioral", mode="lines+markers",
            line=dict(color="#F2B84B", width=2, dash="dash"),
            marker=dict(size=4)
        ))
    
    fig_trend.add_trace(go.Scatter(
        x=v_df["window_start"], y=v_df["dynamic_risk_score"],
        name="Dynamic Risk", mode="lines+markers",
        line=dict(color="#d42c26", width=3),
        marker=dict(size=6)
    ))
    
    if "topological_component" in v_df.columns:
        fig_trend.add_trace(go.Scatter(
            x=v_df["window_start"], y=v_df["topological_component"],
            name="TDA Topo", mode="lines+markers",
            line=dict(color="#6C5DD3", width=2, dash="dot"),
            marker=dict(size=4)
        ))

    fig_trend.update_layout(
        plot_bgcolor="#0E1530", paper_bgcolor="#0E1530", font=dict(color="#EAF0FB"),
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 209, 0.1)", range=[0, 1]),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_g2:
    st.markdown("#### Vendor Risk Radar")
    radar_data = df.groupby('vendor_id')['dynamic_risk_score'].max().reset_index()
    radar_data["risk_tier"] = radar_data["dynamic_risk_score"].apply(lambda s: "High" if s >= 0.65 else ("Medium" if s >= 0.35 else "Low"))
    radar_data = radar_data.sort_values("vendor_id")
    
    node_colors = []
    for idx, row in radar_data.iterrows():
        if row["vendor_id"] == selected_vendor:
            node_colors.append("#FFFFFF") 
        elif row["risk_tier"] == "High":
            node_colors.append("#d42c26")
        elif row["risk_tier"] == "Medium":
            node_colors.append("#fad046")
        else:
            node_colors.append("#3ECF8E")
            
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=radar_data["dynamic_risk_score"], theta=radar_data["vendor_id"],
        fill='toself', fillcolor='rgba(148, 163, 209, 0.08)', 
        line=dict(color='rgba(148, 163, 209, 0.4)', width=1.5), mode='lines+markers',
        marker=dict(color=node_colors, size=12, line=dict(color="#0E1530", width=2))
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False), bgcolor="#0E1530", angularaxis=dict(color="#EAF0FB")),
        paper_bgcolor="#0E1530", font=dict(color="#EAF0FB"), margin=dict(l=30, r=30, t=10, b=10), height=320
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# --- BOTTOM ROW: DETAILS ---
st.markdown("<br>", unsafe_allow_html=True)
col_d1, col_d2 = st.columns([1, 1.8], gap="large")

with col_d1:
    tier_color = "#d42c26" if v_latest["risk_tier"] == "High" else ("#fad046" if v_latest["risk_tier"] == "Medium" else "#3ECF8E")
    
    st.markdown(f"### Risk Status: <span style='background:rgba(255,255,255,0.05); color:{tier_color}; padding:4px 10px; border-radius:6px; font-size:22px; font-family:monospace;'>{selected_vendor}</span>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="panel-box" style="margin-top: 15px; margin-bottom: 25px;">
        <div style="font-size:12px; color:#8A93B8; letter-spacing: 0.5px;">CURRENT DYNAMIC RISK SCORE</div>
        <div style="font-size:42px; font-weight:bold; color:{tier_color}; line-height: 1.2;">{v_latest['dynamic_risk_score']:.3f}</div>
        <div style="font-size:13px; font-weight:bold; color:{tier_color}; text-transform:uppercase;">{v_latest['risk_tier']} RISK TIER</div>
    </div>
    """, unsafe_allow_html=True)
    
    csv_data = v_df.to_csv(index=False).encode('utf-8')
    st.download_button(label=f"Download CSV Data", data=csv_data, file_name=f"{selected_vendor}_risk_report.csv", mime="text/csv")

with col_d2:
    st.markdown("### Investigation Details")
    
    st.markdown("<div style='font-size: 15px; font-weight: 600; margin-top: 15px; margin-bottom: 12px;'>Explanation of Risk Rate:</div>", unsafe_allow_html=True)
    reasons = [r.strip() for r in str(v_latest["risk_reasons"]).split(";") if r.strip()]
    reasons_html = "".join([f'<span class="reason-badge">{r}</span>' for r in reasons])
    st.markdown(f"<div style='margin-bottom: 30px;'>{reasons_html}</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 15px; font-weight: 600; margin-bottom: 12px;'>Recommended Actions:</div>", unsafe_allow_html=True)
    actions = [a.strip() for a in str(v_latest["recommended_actions"]).split(";") if a.strip()]
    for act in actions:
        st.markdown(f"<div style='margin-bottom: 10px; font-size: 15px;'>&bull; &nbsp;⚡ {act}</div>", unsafe_allow_html=True)
        
st.markdown("<br><br>", unsafe_allow_html=True)