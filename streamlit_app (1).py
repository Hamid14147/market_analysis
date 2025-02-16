import streamlit as st

# Install plotly if not present
import subprocess
import sys

def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly.express"])

try:
    import plotly.graph_objects as go
except ImportError:
    install_packages()
    import plotly.graph_objects as go
from market_analyzer import MarketEntryAnalyzer, analyze_markets
import os

# Set page configuration
st.set_page_config(
    page_title="Market Entry Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🌍 Market Entry Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive market entry analysis for different countries, 
including economic metrics, risk assessment, and future forecasts.
""")

# Country selection
available_countries = ["Japan", "Brazil", "France", "Canada"]
selected_countries = st.multiselect(
    "Select Countries to Analyze",
    available_countries,
    default=["Japan", "Brazil"]
)

# Analysis execution
if st.button("Analyze Markets") and selected_countries:
    # Run analysis
    results = analyze_markets(selected_countries)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Market Overview", "🎯 Detailed Analysis", "📊 Visualizations"])
    
    with tab1:
        # Overview metrics
        st.subheader("Market Entry Scores")
        cols = st.columns(len(results))
        for idx, analysis in enumerate(results):
            with cols[idx]:
                st.metric(
                    label=analysis['country'],
                    value=f"{analysis['market_score']}%",
                    delta=f"{analysis['status']}"
                )
    
    with tab2:
        # Detailed country analysis
        for analysis in results:
            with st.expander(f"🎯 {analysis['country']} Detailed Analysis", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Market Overview")
                    for metric, value in analysis['current_metrics'].items():
                        st.metric(label=metric, value=value)
                
                with col2:
                    st.subheader("Risk Assessment")
                    risk = analysis['risk_analysis']
                    st.info(f"Risk Rating: {risk['risk_rating']}")
                    st.progress(risk['total_score']/100)
                    st.write(f"Total Risk Score: {risk['total_score']:.2f}")
                
                # Strengths and Weaknesses
                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("💪 Key Strengths")
                    for strength in analysis['strengths']:
                        st.success(strength)
                
                with col4:
                    st.subheader("⚠️ Key Challenges")
                    for weakness in analysis['weaknesses']:
                        st.warning(weakness)
                
                # Market Outlook
                st.subheader("📈 5-Year Market Outlook")
                forecasts = analysis['forecasts']
                forecast_cols = st.columns(4)
                
                metrics = [
                    ("GDP", "forecasted_value", "growth_rate", "T USD"),
                    ("Population", "forecasted_value", "growth_rate", "M"),
                    ("Consumer_Spending", "forecasted_value", "growth_rate", "T USD"),
                    ("Economic_Growth", "forecasted_value", None, "%")
                ]
                
                for idx, (metric, value_key, growth_key, unit) in enumerate(metrics):
                    with forecast_cols[idx]:
                        metric_data = forecasts[metric]
                        value = metric_data[value_key]
                        if growth_key:
                            growth = metric_data[growth_key]
                            st.metric(
                                label=f"{metric}",
                                value=f"{value:.1f} {unit}",
                                delta=f"{growth:.1f}%"
                            )
                        else:
                            st.metric(
                                label=f"{metric}",
                                value=f"{value:.1f}{unit}"
                            )
    
    with tab3:
        # Visualizations
        st.subheader("📊 Market Comparison Visualizations")
        
        # Market Entry Scores
        fig1 = go.Figure(data=[
            go.Bar(
                x=[r['country'] for r in results],
                y=[r['market_score'] for r in results],
                text=[f"{r['market_score']}%" for r in results],
                textposition='auto',
            )
        ])
        fig1.update_layout(
            title="Market Entry Scores by Country",
            xaxis_title="Country",
            yaxis_title="Score (%)",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Risk Scores
        fig2 = go.Figure(data=[
            go.Bar(
                x=[r['country'] for r in results],
                y=[r['risk_analysis']['total_score'] for r in results],
                text=[f"{r['risk_analysis']['total_score']:.1f}" for r in results],
                textposition='auto',
            )
        ])
        fig2.update_layout(
            title="Risk Scores by Country (Lower is Better)",
            xaxis_title="Country",
            yaxis_title="Risk Score",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Please select countries and click 'Analyze Markets' to begin the analysis.")
