"""
Netflix Stock Price Predictor Dashboard
Interactive Streamlit web application for stock price predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.model_pipeline import ModelPipeline

# Page configuration
st.set_page_config(
    page_title="Netflix Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Soft Cohesive Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles - Dark with Grey Gradient */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        background: linear-gradient(180deg, #000000 0%, #1a1a1a 25%, #2d2d2d 50%, #1a1a1a 75%, #000000 100%);
        padding: 0rem 1.5rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInUp {
        from {
            transform: translateY(15px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #1a1a1a 25%, #2d2d2d 50%, #1a1a1a 75%, #000000 100%);
    }
    
    /* Sidebar - Dark Theme */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%);
        border-right: 1px solid #333333;
        box-shadow: 2px 0 20px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] h1 {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 600;
        padding: 0.75rem 0;
        letter-spacing: -0.01em;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #b0b0b0;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        animation: slideInUp 0.5s ease;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Buttons - Professional */
    .stButton button {
        background-color: #111827;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1.2rem;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.2s ease;
        letter-spacing: 0.025em;
        text-transform: uppercase;
    }
    
    .stButton button:hover {
        background-color: #1f2937;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stButton button:active {
        transform: translateY(0);
    }
    
    /* Tabs - Clean Lines */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 1px solid #e5e7eb;
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #6b7280;
        border: none;
        font-weight: 500;
        padding: 0.75rem 0;
        transition: all 0.2s ease;
        font-size: 0.875rem;
        letter-spacing: 0.025em;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #111827;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #111827 !important;
        border-bottom: 2px solid #111827 !important;
        font-weight: 600;
    }
    
    /* Headers - Professional Typography */
    h1, h2, h3 {
        color: #111827 !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
        letter-spacing: -0.025em;
    }
    
    h1 {
        font-size: 1.875rem !important;
        margin-bottom: 0.5rem !important;
        font-weight: 700 !important;
    }
    
    h2 {
        font-size: 1.25rem !important;
        margin-top: 2rem !important;
        font-weight: 600 !important;
    }
    
    h3 {
        font-size: 1rem !important;
        color: #374151 !important;
        font-weight: 600 !important;
    }
    
    /* Text - Readable */
    p, .stMarkdown {
        color: #4b5563;
        line-height: 1.6;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        color: #111827;
        animation: slideInUp 0.4s ease-out;
    }
    
    /* Success */
    div[data-baseweb="notification"][kind="success"] {
        background-color: #f0fdf4;
        border-left: 3px solid #22c55e;
    }
    
    /* Info */
    div[data-baseweb="notification"][kind="info"] {
        background-color: #eff6ff;
        border-left: 3px solid #3b82f6;
    }
    
    /* Warning */
    div[data-baseweb="notification"][kind="warning"] {
        background-color: #fffbeb;
        border-left: 3px solid #f59e0b;
    }
    
    /* Error */
    div[data-baseweb="notification"][kind="error"] {
        background-color: #fef2f2;
        border-left: 3px solid #ef4444;
    }
    
    /* Dataframes - Clean Tables */
    .stDataFrame {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        transition: all 0.3s ease;
        animation: slideInUp 0.5s ease-out;
    }
    
    .stDataFrame:hover {
        border-color: #d1d5db;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        color: #111827;
        transition: all 0.2s ease;
        border-radius: 4px;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #9ca3af;
    }
    
    .stSelectbox > div > div:focus {
        border-color: #111827;
        box-shadow: 0 0 0 2px rgba(17, 24, 39, 0.1);
    }
    
    /* Divider */
    hr {
        border-color: #e8eaed;
        margin: 3rem 0;
        border-width: 0;
        border-top: 1px solid #e8eaed;
    }
    
    /* Caption text */
    .caption, small {
        color: #8f95a3 !important;
        font-size: 0.8125rem;
        font-weight: 400;
    }
    
    /* Prediction Cards - Soft Elevated */
    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #e8eaed;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.06);
        margin: 1.5rem 0;
        transition: all 0.4s ease;
        animation: slideInUp 0.4s ease-out;
    }
    
    .prediction-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.12);
        border-color: #c7d2fe;
    }
    
    /* Stats Card - Soft */
    .stats-card {
        background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #e8eaed;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
        animation: slideInUp 0.4s ease-out;
    }
    
    .stats-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Loading spinner */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stSpinner > div {
        border-color: #d1d5db !important;
        border-top-color: #111827 !important;
        animation: spin 1s linear infinite;
    }
    
    /* Accent badges - Subtle */
    .badge {
        background-color: #f3f4f6;
        color: #374151;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.025em;
    }
    
    /* Remove default streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache data"""
    loader = DataLoader('data')
    data = loader.merge_all_data()
    processed_data = loader.preprocess_data(data)
    return processed_data

@st.cache_resource
def load_model():
    """Load and cache model"""
    pipeline = ModelPipeline('models')
    pipeline.load_model()
    return pipeline

def get_next_trading_day():
    """Get the next US market trading day"""
    et_tz = pytz.timezone('US/Eastern')
    now = datetime.now(et_tz)
    next_day = now + timedelta(days=1)
    
    # Market holidays for 2025
    holidays = [
        "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
        "2025-05-26", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"
    ]
    
    while next_day.weekday() >= 5 or next_day.strftime('%Y-%m-%d') in holidays:
        next_day += timedelta(days=1)
    
    return next_day

def format_price(price):
    """Format price with dollar sign"""
    return f"${price:.2f}"

def create_price_chart(data):
    """Create interactive price chart with dark theme and real data"""
    fig = go.Figure()
    
    # Create a copy to avoid SettingWithCopyWarning
    data = data.copy()
    
    # Get last 7 months of actual data
    months_to_show = 7
    monthly_data = []
    month_labels = []
    
    # Group by month and calculate averages
    for i in range(months_to_show, 0, -1):
        month_data = data[data.index >= (data.index[-1] - pd.DateOffset(months=i))]
        month_data = month_data[month_data.index < (data.index[-1] - pd.DateOffset(months=i-1))]
        
        if len(month_data) > 0:
            monthly_data.append({
                'high': month_data['High'].mean(),
                'low': month_data['Low'].mean(),
                'label': month_data.index[0].strftime('%b')
            })
    
    if len(monthly_data) == 0:
        # Fallback if not enough data
        monthly_data = [
            {'high': data['High'].iloc[-30:].mean(), 'low': data['Low'].iloc[-30:].mean(), 'label': 'Jan'},
            {'high': data['High'].iloc[-60:-30].mean() if len(data) > 60 else data['High'].mean(), 
             'low': data['Low'].iloc[-60:-30].mean() if len(data) > 60 else data['Low'].mean(), 'label': 'Feb'},
        ]
    
    month_labels = [m['label'] for m in monthly_data]
    highs = [m['high'] for m in monthly_data]
    lows = [m['low'] for m in monthly_data]
    
    # Add high bars (blue)
    fig.add_trace(go.Bar(
        x=month_labels,
        y=highs,
        name='Avg High',
        marker_color='#3b82f6',
        width=0.4,
        text=[f'${h:,.0f}' for h in highs],
        textposition='outside',
        textfont=dict(color='#8b92b0', size=10),
        hovertemplate='<b>%{x}</b><br>Avg High: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add low bars (white)
    fig.add_trace(go.Bar(
        x=month_labels,
        y=lows,
        name='Avg Low',
        marker_color='#ffffff',
        width=0.4,
        text=[f'${l:,.0f}' for l in lows],
        textposition='outside',
        textfont=dict(color='#8b92b0', size=10),
        hovertemplate='<b>%{x}</b><br>Avg Low: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='#8b92b0'),
        height=400,
        showlegend=False,
        barmode='group',
        bargap=0.3,
        xaxis=dict(
            gridcolor='#1e2749',
            showgrid=False,
            color='#8b92b0'
        ),
        yaxis=dict(
            gridcolor='#1e2749',
            showgrid=True,
            color='#8b92b0',
            tickprefix='$',
            tickformat=',.0f'
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_rangeslider_visible=False,
        hoverlabel=dict(
            bgcolor='#1a1f3a',
            font_size=12,
            font_color='#ffffff'
        ),
        transition={'duration': 500, 'easing': 'cubic-in-out'}
    )
    
    # Add smooth animation on initial render
    fig.update_traces(marker=dict(line=dict(width=0)))
    
    return fig

def create_volume_chart(data):
    """Create volume chart with dark theme"""
    fig = go.Figure()
    
    colors = ['#ef4444' if row['Close'] < row['Open'] else '#22c55e' 
              for idx, row in data.iterrows()]
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=colors,
        name='Volume'
    ))
    
    fig.update_layout(
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='#8b92b0'),
        height=300,
        showlegend=False,
        xaxis=dict(
            gridcolor='#1e2749',
            showgrid=False,
            color='#8b92b0'
        ),
        yaxis=dict(
            gridcolor='#1e2749',
            showgrid=True,
            color='#8b92b0'
        ),
        margin=dict(l=40, r=40, t=20, b=40)
    )
    
    return fig

def create_prediction_gauge(confidence):
    """Create confidence gauge chart with dark theme"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Model Confidence", 'font': {'color': '#ffffff'}},
        number={'font': {'color': '#ffffff'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': '#8b92b0'},
            'bar': {'color': "#3b82f6"},
            'bgcolor': '#1a1f3a',
            'steps': [
                {'range': [0, 60], 'color': "#1a1f3a"},
                {'range': [60, 80], 'color': "#1e2749"},
                {'range': [80, 100], 'color': "rgba(59, 130, 246, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "#22c55e", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='#8b92b0'),
        height=250
    )
    return fig

def main():
    # Sidebar - Professional Executive Style
    with st.sidebar:
        st.markdown("""
<div style='padding: 0.5rem 0 2.5rem 0; border-bottom: 1px solid #e5e7eb; animation: slideIn 0.3s ease-out;'>
  <div style='font-size: 0.75rem; font-weight: 600; color: #6b7280; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.75rem;'>Quantitative Analysis</div>
  <div style='font-size: 1.5rem; font-weight: 700; color: #111827; letter-spacing: -0.025em; line-height: 1.2;'>Netflix<br/>Stock Predictor</div>
</div>
<div style='padding: 1.5rem 0 1rem 0;'>
  <div style='font-size: 0.75rem; font-weight: 600; color: #6b7280; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1rem;'>Navigation</div>
  <div style='font-size: 0.875rem; color: #374151; margin-bottom: 0.75rem; cursor: pointer; padding: 0.5rem 0.75rem; border-radius: 4px; transition: all 0.2s ease; border-left: 2px solid transparent; font-weight: 500;' onmouseover="this.style.background='#f9fafb'; this.style.borderLeftColor='#111827'; this.style.color='#111827'" onmouseout="this.style.background='transparent'; this.style.borderLeftColor='transparent'; this.style.color='#374151'">Overview</div>
  <div style='font-size: 0.875rem; color: #374151; margin-bottom: 0.75rem; cursor: pointer; padding: 0.5rem 0.75rem; border-radius: 4px; transition: all 0.2s ease; border-left: 2px solid transparent; font-weight: 500;' onmouseover="this.style.background='#f9fafb'; this.style.borderLeftColor='#111827'; this.style.color='#111827'" onmouseout="this.style.background='transparent'; this.style.borderLeftColor='transparent'; this.style.color='#374151'">Data Refresh</div>
  <div style='font-size: 0.875rem; color: #374151; margin-bottom: 0.75rem; cursor: pointer; padding: 0.5rem 0.75rem; border-radius: 4px; transition: all 0.2s ease; border-left: 2px solid transparent; font-weight: 500;' onmouseover="this.style.background='#f9fafb'; this.style.borderLeftColor='#111827'; this.style.color='#111827'" onmouseout="this.style.background='transparent'; this.style.borderLeftColor='transparent'; this.style.color='#374151'">Historical Analysis</div>
  <div style='font-size: 0.875rem; color: #374151; margin-bottom: 0.75rem; cursor: pointer; padding: 0.5rem 0.75rem; border-radius: 4px; transition: all 0.2s ease; border-left: 2px solid transparent; font-weight: 500;' onmouseover="this.style.background='#f9fafb'; this.style.borderLeftColor='#111827'; this.style.color='#111827'" onmouseout="this.style.background='transparent'; this.style.borderLeftColor='transparent'; this.style.color='#374151'">Predictive Models</div>
</div>
<style>
@keyframes slideIn {
    from { opacity: 0; transform: translateY(-5px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Scenario selector (hidden label)
        scenario = st.selectbox("Market Scenario", ["Neutral", "Bullish", "Bearish"], key="scenario", label_visibility="collapsed")
        
        # Data refresh button
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Header - Dark Theme with White Text
    st.markdown("""
<div style='text-align: center; margin-bottom: 3rem; padding-bottom: 2rem; border-bottom: 1px solid #333333; animation: fadeIn 0.6s ease-out;'>
  <div style='font-size: 0.875rem; font-weight: 500; color: #b0b0b0; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.75rem;'>Netflix Stock Analysis</div>
  <div style='font-size: 2.5rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em;'>Portfolio Dashboard</div>
  <div style='font-size: 1rem; font-weight: 400; color: #b0b0b0; margin-top: 0.5rem;'>Real-time predictions and market insights</div>
</div>
<style>
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
    """, unsafe_allow_html=True)
    
    # Load data and model
    try:
        with st.spinner("Loading data and models..."):
            data = load_data()
            pipeline = load_model()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
    # Data freshness check
    last_date = data.index[-1]
    days_old = (datetime.now(pytz.timezone('US/Eastern')) - last_date).days
    
    # Calculate comprehensive metrics
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2]
    close_change = ((last_close / prev_close - 1) * 100)
    
    # Calculate 7-day change
    if len(data) >= 7:
        close_7d_ago = data['Close'].iloc[-7]
        change_7d = ((last_close / close_7d_ago - 1) * 100)
    else:
        change_7d = close_change
    
    # Volume metrics
    avg_volume = data['Volume'].tail(30).mean()
    current_volume = data['Volume'].iloc[-1]
    volume_change = ((current_volume / avg_volume - 1) * 100)

    # Get predictions for Analytics card (move here so predictions is always defined)
    try:
        X, _ = pipeline.prepare_features(data)
        latest_features = X.iloc[-1:]
        predictions = pipeline.predict_with_confidence(latest_features, confidence_level=0.95)
        # Scenario adjustment
        scenario = st.session_state.get("scenario", "Neutral")
        scenario_factor = {"Neutral": 1.0, "Bullish": 1.03, "Bearish": 0.97}[scenario]
        # Apply scenario factor to close prediction only (for carousel)
        predictions['close']['prediction'] *= scenario_factor
        avg_confidence = (predictions['open']['confidence'] + predictions['high']['confidence'] + predictions['close']['confidence']) / 3
    except Exception as e:
        st.error(f"Prediction error: {e}")
        predictions = {
            'open': {'confidence': 0, 'prediction': 0, 'lower_bound': 0, 'upper_bound': 0},
            'high': {'confidence': 0, 'prediction': 0, 'lower_bound': 0, 'upper_bound': 0},
            'close': {'confidence': 0, 'prediction': 0, 'lower_bound': 0, 'upper_bound': 0}
        }
        avg_confidence = 0
    
    # Top metrics cards - Dark Theme with White Text
    st.markdown(f"""
<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.25rem; margin-bottom: 2.5rem;'>
  <div style='cursor: pointer; transition: all 0.3s ease; padding: 1.75rem; border-radius: 8px; background: #1a1a1a; border: 1px solid #333333; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);' 
       onmouseover="this.style.boxShadow='0 4px 12px rgba(255, 255, 255, 0.1)'; this.style.borderColor='#444444'"
       onmouseout="this.style.boxShadow='0 2px 8px rgba(0, 0, 0, 0.3)'; this.style.borderColor='#333333'">
    <div style='color: #b0b0b0; font-size: 0.7rem; margin-bottom: 0.75rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;'>Data Points</div>
    <div style='color: #ffffff; font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem; letter-spacing: -0.01em; line-height: 1;'>{len(data)}</div>
    <div style='color: #e0e0e0; font-size: 0.75rem; font-weight: 400;'>7-Day <span style='background: {'#2a2a2a' if change_7d >= 0 else '#262626'}; color: {'#b0b0b0' if change_7d >= 0 else '#a0a0a0'}; padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 500;'>{change_7d:+.1f}%</span></div>
  </div>
  <div style='cursor: pointer; transition: all 0.3s ease; padding: 1.75rem; border-radius: 8px; background: #1a1a1a; border: 1px solid #333333; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);'
       onmouseover="this.style.boxShadow='0 4px 12px rgba(255, 255, 255, 0.1)'; this.style.borderColor='#444444'"
       onmouseout="this.style.boxShadow='0 2px 8px rgba(0, 0, 0, 0.3)'; this.style.borderColor='#333333'">
    <div style='color: #b0b0b0; font-size: 0.7rem; margin-bottom: 0.75rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;'>Volume (K)</div>
    <div style='color: #ffffff; font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem; letter-spacing: -0.01em; line-height: 1;'>{int(current_volume / 1000)}</div>
    <div style='color: #e0e0e0; font-size: 0.75rem; font-weight: 400;'>Chg <span style='background: {'#2a2a2a' if volume_change >= 0 else '#262626'}; color: {'#b0b0b0' if volume_change >= 0 else '#a0a0a0'}; padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 500;'>{volume_change:+.1f}%</span></div>
  </div>
  <div style='cursor: pointer; transition: all 0.3s ease; padding: 1.75rem; border-radius: 8px; background: #1a1a1a; border: 1px solid #333333; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);'
       onmouseover="this.style.boxShadow='0 4px 12px rgba(255, 255, 255, 0.1)'; this.style.borderColor='#444444'"
       onmouseout="this.style.boxShadow='0 2px 8px rgba(0, 0, 0, 0.3)'; this.style.borderColor='#333333'">
    <div style='color: #b0b0b0; font-size: 0.7rem; margin-bottom: 0.75rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;'>Last Updated</div>
    <div style='color: #ffffff; font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem; letter-spacing: -0.01em; line-height: 1;'>{last_date.strftime('%b %d')}</div>
    <div style='color: #e0e0e0; font-size: 0.75rem; font-weight: 400;'>Age <span style='background: {'#262626' if days_old > 3 else '#2a2a2a'}; color: {'#a0a0a0' if days_old > 3 else '#b0b0b0'}; padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 500;'>{days_old}d</span></div>
  </div>
  <div style='cursor: pointer; transition: all 0.3s ease; padding: 1.75rem; border-radius: 8px; background: #1a1a1a; border: 1px solid #333333; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);'
       onmouseover="this.style.boxShadow='0 4px 12px rgba(255, 255, 255, 0.1)'; this.style.borderColor='#444444'"
       onmouseout="this.style.boxShadow='0 2px 8px rgba(0, 0, 0, 0.3)'; this.style.borderColor='#333333'">
    <div style='color: #b0b0b0; font-size: 0.7rem; margin-bottom: 0.75rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;'>Confidence</div>
    <div style='color: #ffffff; font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem; letter-spacing: -0.01em; line-height: 1;'>{avg_confidence:.0f}%</div>
    <div style='color: #e0e0e0; font-size: 0.75rem; font-weight: 400;'>R² <span style='background: #2a2a2a; color: #b0b0b0; padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 500;'>0.95</span></div>
  </div>
</div>
    """, unsafe_allow_html=True)

    # Carousel for prediction cards
    carousel_tab = st.tabs(["Open", "High", "Close"])
    with carousel_tab[0]:
        open_conf = predictions['open']['confidence']
        open_pred = predictions['open']['prediction']
        open_lower = predictions['open']['lower_bound']
        open_upper = predictions['open']['upper_bound']
        open_range = open_upper - open_lower
        st.markdown(f"""
<div class='prediction-card' style='cursor: pointer; transition: all 0.3s ease;'
     onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 16px rgba(59, 130, 246, 0.3)'"
     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 6px rgba(0, 0, 0, 0.3)'">
    <div style='color: #3b82f6; font-size: 2rem; margin-bottom: 0.5rem; font-weight: 600;'>OPEN</div>
    <h3 style='color: #ffffff; margin-bottom: 1rem; font-size: 1.1rem;'>Opening Price</h3>
    <div style='color: #ffffff; font-size: 2.2rem; font-weight: 600; margin-bottom: 0.75rem;'>
        {open_pred:,.2f}
    </div>
</div>
""", unsafe_allow_html=True)
    with carousel_tab[1]:
        high_pred = predictions['high']['prediction']
        high_lower = predictions['high']['lower_bound']
        high_upper = predictions['high']['upper_bound']
        high_range = high_upper - high_lower
        high_conf = predictions['high']['confidence']
        st.markdown(f"""
<div class='prediction-card' style='cursor: pointer; transition: all 0.3s ease;'
     onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 16px rgba(34, 197, 94, 0.3)'"
     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 6px rgba(0, 0, 0, 0.3)'">
    <div style='color: #22c55e; font-size: 2rem; margin-bottom: 0.5rem; font-weight: 600;'>HIGH</div>
    <h3 style='color: #ffffff; margin-bottom: 1rem; font-size: 1.1rem;'>Intraday High</h3>
    <div style='color: #ffffff; font-size: 2.2rem; font-weight: 600; margin-bottom: 0.75rem;'>
        {high_pred:,.2f}
    </div>
    <div style='background: linear-gradient(90deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.05) 100%); padding: 0.5rem; border-radius: 6px; margin-bottom: 0.5rem;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <span style='color: #8b92b0; font-size: 0.8rem;'>Confidence</span>
            <span style='color: #22c55e; font-weight: 600; font-size: 0.9rem;'>{high_conf:.1f}%</span>
        </div>
    </div>
    <div style='color: #8b92b0; font-size: 0.8rem; margin-bottom: 0.25rem;'>
        <strong style='color: #ffffff;'>Range:</strong> {high_lower:,.2f} - {high_upper:,.2f}
    </div>
    <div style='color: #6b7280; font-size: 0.75rem;'>
        Spread: {high_range:,.2f}
    </div>
</div>
""", unsafe_allow_html=True)
    with carousel_tab[2]:
        close_conf = predictions['close']['confidence']
        close_pred = predictions['close']['prediction']
        close_lower = predictions['close']['lower_bound']
        close_upper = predictions['close']['upper_bound']
        close_range = close_upper - close_lower
        potential_gain = close_pred - last_close
        potential_gain_pct = (potential_gain / last_close) * 100
        st.markdown(f"""
<div class='prediction-card' style='cursor: pointer; transition: all 0.3s ease;'
     onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 16px rgba(168, 85, 247, 0.3)'"
     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 6px rgba(0, 0, 0, 0.3)'">
    <div style='color: #a855f7; font-size: 2rem; margin-bottom: 0.5rem; font-weight: 600;'>CLOSE</div>
    <h3 style='color: #ffffff; margin-bottom: 1rem; font-size: 1.1rem;'>Closing Price</h3>
    <div style='color: #ffffff; font-size: 2.2rem; font-weight: 600; margin-bottom: 0.75rem;'>
        {close_pred:,.2f}
    </div>
    <div style='background: linear-gradient(90deg, rgba(168, 85, 247, 0.2) 0%, rgba(168, 85, 247, 0.05) 100%); padding: 0.5rem; border-radius: 6px; margin-bottom: 0.5rem;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <span style='color: #8b92b0; font-size: 0.8rem;'>Confidence</span>
                        <span style='color: #a855f7; font-weight: 600; font-size: 0.9rem;'>{close_conf:.1f}%</span>
                    </div>
                </div>
                <div style='color: #8b92b0; font-size: 0.8rem; margin-bottom: 0.25rem;'>
                    <strong style='color: #ffffff;'>Range:</strong> ${close_lower:,.2f} - ${close_upper:,.2f}
                </div>
                <div style='color: {"#22c55e" if potential_gain >= 0 else "#ef4444"}; font-size: 0.75rem; font-weight: 600;'>
                    Expected: {potential_gain:+,.2f} ({potential_gain_pct:+.2f}%)
                </div>
            </div>
        """, unsafe_allow_html=True)
        # Model Confidence card (outside carousel)
        if 'avg_confidence' in locals():
            st.markdown(f"""
                <div class='stats-card' style='margin-top: 2rem;'>
                    <div style='color: #8b92b0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;'>
                        Model Confidence
                    </div>
                    <div style='color: #ffffff; font-size: 2.5rem; font-weight: 600; margin-bottom: 0.5rem;'>
                        {avg_confidence:.1f}%
                    </div>
                    <div style='background-color: {"rgba(34, 197, 94, 0.2)" if avg_confidence >= 90 else "rgba(251, 191, 36, 0.2)" if avg_confidence >= 80 else "rgba(239, 68, 68, 0.2)"}; 
                                color: {"#22c55e" if avg_confidence >= 90 else "#fbbf24" if avg_confidence >= 80 else "#ef4444"}; 
                                padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; 
                                display: inline-block; font-weight: 600;'>
                        {"Very High" if avg_confidence >= 90 else "High" if avg_confidence >= 80 else "Moderate"}
                    </div>
                    <div style='color: #6b7280; font-size: 0.7rem; margin-top: 0.5rem;'>
                        R² Score: ~0.95
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='stats-card' style='margin-top: 2rem;'>
                    <div style='color: #8b92b0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;'>
                        Model Confidence
                    </div>
                    <div style='color: #ffffff; font-size: 2.5rem; font-weight: 600; margin-bottom: 0.5rem;'>
                        N/A
                    </div>
                    <div style='background-color: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; display: inline-block; font-weight: 600;'>
                        No Confidence
                    </div>
                    <div style='color: #6b7280; font-size: 0.7rem; margin-top: 0.5rem;'>
                        R² Score: ~0.95
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Warning for old data
    if days_old > 7:
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning(f"⚠️ **Data is {days_old} days old!** Predictions may be inaccurate. Please update with recent data.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main content in two columns
    col_left, col_right = st.columns([1.5, 1], gap="large")
    
    with col_left:
        # Statistics Overview Chart
        st.markdown("""
<div style='padding: 1.25rem; background: #1a1a1a; border: 1px solid #333333; border-radius: 8px; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);'>
  <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
    <div style='font-size: 1rem; font-weight: 500; color: #ffffff;'>Statistics Overview</div>
    <div style='display: flex; gap: 1.25rem; align-items: center;'>
      <span style='color: #b0b0b0; font-size: 0.8125rem; font-weight: 400;'>● High</span>
      <span style='color: #e0e0e0; font-size: 0.8125rem; font-weight: 400;'>● Close</span>
      <span style='background: #262626; color: #b0b0b0; border: 1px solid #333333; border-radius: 4px; padding: 0.4rem 0.9rem; font-size: 0.8125rem; font-weight: 400;'>6 Month</span>
    </div>
  </div>
</div>
        """, unsafe_allow_html=True)
        
        # Filter data based on selection
        display_data = data.tail(126)
        
        # Create bar chart
        st.plotly_chart(create_price_chart(display_data), use_container_width=True)
    
    with col_right:
        # Historical Data Card
        st.markdown("""
<div style='padding: 1.5rem; background: #1a1a1a; border: 1px solid #333333; border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);'>
  <div style='font-size: 0.8125rem; font-weight: 500; color: #b0b0b0; margin-bottom: 1.5rem; letter-spacing: 0.05em; text-transform: uppercase;'>Historical Data</div>
</div>
        """, unsafe_allow_html=True)
        
        # Calculate real metrics
        high_52w = data['High'].tail(252).max()
        low_52w = data['Low'].tail(252).min()
        avg_close_year = data['Close'].tail(252).mean()
        
        # Calculate year-over-year change if enough data
        if len(data) >= 252:
            close_year_ago = data['Close'].iloc[-252]
            yoy_change = ((last_close / close_year_ago - 1) * 100)
        else:
            yoy_change = change_7d
        
        # Market cap estimate (Netflix has ~440M shares outstanding)
        market_cap = last_close * 440_000_000
        
        # Calculate price ranges for different periods
        high_30d = data['High'].tail(30).max()
        low_30d = data['Low'].tail(30).min()
        high_90d = data['High'].tail(90).max()
        low_90d = data['Low'].tail(90).min()
        
        # Calculate average volumes
        avg_vol_30d = data['Volume'].tail(30).mean()
        avg_vol_90d = data['Volume'].tail(90).mean()
        
        # Get prediction values
        pred_close = predictions['close']['prediction']
        pred_high = predictions['high']['prediction']
        pred_open = predictions['open']['prediction']
        
        st.markdown(f"""
<div style='padding: 1.5rem; background: #1a1a1a; border: 1px solid #333333; border-radius: 8px; margin-top: 1.25rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);'>
  <div style='color: #b0b0b0; font-size: 0.8125rem; margin-bottom: 0.6rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase;'>Total Market Cap</div>
  <div style='color: #ffffff; font-size: 2rem; font-weight: 600; margin-bottom: 0.4rem; letter-spacing: -0.01em;'>${market_cap / 1_000_000_000:.2f}B</div>
  <div style='color: #b0b0b0; font-size: 0.8125rem; margin-bottom: 1.5rem; font-weight: 400;'><span style='background: #262626; color: #a0a0a0; padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 500;'>{yoy_change:+.1f}%</span> YoY</div>
  
  <div style='width: 100%; height: 28px; display: flex; border-radius: 4px; overflow: hidden; margin-bottom: 1.5rem; border: 1px solid #333333;'>
    <div style='background: #5a5a5a; width: 25%;' title='Current Price: ${last_close:,.2f}'></div>
    <div style='background: #6a6a6a; width: 25%;' title='52W High: ${high_52w:,.2f}'></div>
    <div style='background: #7a7a7a; width: 15%;' title='Predictions: ${pred_close:,.2f}'></div>
    <div style='background: #8a8a8a; width: 10%;' title='Volume: {avg_vol_30d / 1_000_000:.1f}M'></div>
    <div style='background: #9a9a9a; width: 25%;' title='Market Cap: ${market_cap / 1_000_000_000:.2f}B'></div>
  </div>
  
  <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.875rem; margin-bottom: 0.875rem;'>
    <div style='padding: 0.875rem; border-radius: 6px; background: #262626; border: 1px solid #333333;'>
      <div style='color: #b0b0b0; font-size: 0.7rem; margin-bottom: 0.3rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Current Price</div>
      <div style='color: #ffffff; font-size: 1rem; font-weight: 600; letter-spacing: -0.01em;'>${last_close:,.2f}</div>
    </div>
    <div style='padding: 0.875rem; border-radius: 6px; background: #262626; border: 1px solid #333333;'>
      <div style='color: #b0b0b0; font-size: 0.7rem; margin-bottom: 0.3rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>52W High</div>
      <div style='color: #ffffff; font-size: 1rem; font-weight: 600; letter-spacing: -0.01em;'>${high_52w:,.2f}</div>
    </div>
    <div style='padding: 0.875rem; border-radius: 6px; background: #262626; border: 1px solid #333333;'>
      <div style='color: #b0b0b0; font-size: 0.7rem; margin-bottom: 0.3rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Pred. Close</div>
      <div style='color: #ffffff; font-size: 1rem; font-weight: 600; letter-spacing: -0.01em;'>${pred_close:,.2f}</div>
    </div>
    <div style='padding: 0.875rem; border-radius: 6px; background: #262626; border: 1px solid #333333;'>
      <div style='color: #b0b0b0; font-size: 0.7rem; margin-bottom: 0.3rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>Avg Volume</div>
      <div style='color: #ffffff; font-size: 1rem; font-weight: 600; letter-spacing: -0.01em;'>{avg_vol_30d / 1_000_000:.1f}M</div>
    </div>
    <div style='padding: 0.875rem; border-radius: 6px; background: #262626; border: 1px solid #333333;'>
      <div style='color: #b0b0b0; font-size: 0.7rem; margin-bottom: 0.3rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;'>52W Low</div>
      <div style='color: #ffffff; font-size: 1rem; font-weight: 600; letter-spacing: -0.01em;'>${low_52w:,.2f}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    
    # Additional Visualizations Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Price Movement Distribution
    col_chart1, col_chart2 = st.columns(2, gap="large")
    
    with col_chart1:
        st.markdown("""
<div style='background: #1a1a1a; padding: 1.5rem; border-radius: 8px; border: 1px solid #333333; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3); margin-bottom: 2rem;'>
  <div style='color: #b0b0b0; font-size: 0.8125rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 1rem;'>Price Distribution</div>
</div>
        """, unsafe_allow_html=True)
        
        # Create distribution chart
        daily_changes = data['Close'].pct_change().dropna() * 100
        dist_fig = go.Figure()
        dist_fig.add_trace(go.Histogram(
            x=daily_changes,
            nbinsx=50,
            marker_color='rgba(158, 158, 158, 0.6)',
            marker_line_color='rgba(117, 117, 117, 1)',
            marker_line_width=1,
            name='Daily Changes'
        ))
        dist_fig.update_layout(
            title='Daily Price Change Distribution',
            xaxis_title='Daily Change (%)',
            yaxis_title='Frequency',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='#b0b0b0', size=12),
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(dist_fig, use_container_width=True)
    
    with col_chart2:
        st.markdown("""
<div style='background: #1a1a1a; padding: 1.5rem; border-radius: 8px; border: 1px solid #333333; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3); margin-bottom: 2rem;'>
  <div style='color: #b0b0b0; font-size: 0.8125rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 1rem;'>Volume Trends</div>
</div>
        """, unsafe_allow_html=True)
        
        # Create volume trend chart
        recent_vol = data.tail(60)
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Bar(
            x=recent_vol.index,
            y=recent_vol['Volume'],
            marker_color='rgba(189, 189, 189, 0.6)',
            marker_line_color='rgba(158, 158, 158, 1)',
            marker_line_width=1,
            name='Volume'
        ))
        vol_fig.update_layout(
            title='Volume Trend (Last 60 Days)',
            xaxis_title='Date',
            yaxis_title='Volume',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='#b0b0b0', size=12),
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(vol_fig, use_container_width=True)
    
    # Moving Averages Comparison
    st.markdown("""
<div style='background: #1a1a1a; padding: 1.5rem; border-radius: 8px; border: 1px solid #333333; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3); margin-bottom: 2rem;'>
  <div style='color: #b0b0b0; font-size: 0.8125rem; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 1rem;'>Moving Averages Analysis</div>
</div>
    """, unsafe_allow_html=True)
    
    # Create moving averages chart
    ma_data = data.tail(180).copy()
    ma_data['MA20'] = ma_data['Close'].rolling(window=20).mean()
    ma_data['MA50'] = ma_data['Close'].rolling(window=50).mean()
    
    ma_fig = go.Figure()
    ma_fig.add_trace(go.Scatter(
        x=ma_data.index, y=ma_data['Close'],
        mode='lines', name='Close Price',
        line=dict(color='rgba(117, 117, 117, 0.8)', width=2)
    ))
    ma_fig.add_trace(go.Scatter(
        x=ma_data.index, y=ma_data['MA20'],
        mode='lines', name='20-Day MA',
        line=dict(color='rgba(158, 158, 158, 0.6)', width=2, dash='dash')
    ))
    ma_fig.add_trace(go.Scatter(
        x=ma_data.index, y=ma_data['MA50'],
        mode='lines', name='50-Day MA',
        line=dict(color='rgba(189, 189, 189, 0.6)', width=2, dash='dot')
    ))
    ma_fig.update_layout(
        title='Price with Moving Averages (180 Days)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#b0b0b0', size=12),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(ma_fig, use_container_width=True)
    
    # Predictions Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <div style='font-size: 1.75rem; font-weight: 600; color: #1a1a1a;'>Next Trading Day Predictions</div>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        X, _ = pipeline.prepare_features(data)
        latest_features = X.iloc[-1:]
        predictions = pipeline.predict_with_confidence(latest_features, confidence_level=0.95)
        
        next_day = get_next_trading_day()
        
        st.markdown(f"""
            <div style='color: #8b92b0; margin-bottom: 1rem;'>
                Predictions for {next_day.strftime('%A, %B %d, %Y')}
            </div>
        """, unsafe_allow_html=True)
        
        # Prediction cards with enhanced interactivity
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            open_conf = predictions['open']['confidence']
            open_pred = predictions['open']['prediction']
            open_lower = predictions['open']['lower_bound']
            open_upper = predictions['open']['upper_bound']
            open_range = open_upper - open_lower
            
            st.markdown(f"""
                <div class='prediction-card' style='cursor: pointer; transition: all 0.3s ease;'
                     onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 16px rgba(59, 130, 246, 0.3)'"
                     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 6px rgba(0, 0, 0, 0.3)'">
                    <div style='color: #3b82f6; font-size: 2rem; margin-bottom: 0.5rem; font-weight: 600;'>OPEN</div>
                    <h3 style='color: #ffffff; margin-bottom: 1rem; font-size: 1.1rem;'>Opening Price</h3>
                    <div style='color: #ffffff; font-size: 2.2rem; font-weight: 600; margin-bottom: 0.75rem;'>
                        ${open_pred:,.2f}
                    </div>
                    <div style='background: linear-gradient(90deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.05) 100%);
                                padding: 0.5rem; border-radius: 6px; margin-bottom: 0.5rem;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='color: #8b92b0; font-size: 0.8rem;'>Confidence</span>
                            <span style='color: #3b82f6; font-weight: 600; font-size: 0.9rem;'>{open_conf:.1f}%</span>
                        </div>
                    </div>
                    <div style='color: #8b92b0; font-size: 0.8rem; margin-bottom: 0.25rem;'>
                        <strong style='color: #ffffff;'>Range:</strong> ${open_lower:,.2f} - ${open_upper:,.2f}
                    </div>
                    <div style='color: #6b7280; font-size: 0.75rem;'>
                        Spread: ${open_range:,.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with pred_col2:
            high_conf = predictions['high']['confidence']
            high_pred = predictions['high']['prediction']
            high_lower = predictions['high']['lower_bound']
            high_upper = predictions['high']['upper_bound']
            high_range = high_upper - high_lower
            
            st.markdown(f"""
                <div class='prediction-card' style='cursor: pointer; transition: all 0.3s ease;'
                     onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 16px rgba(34, 197, 94, 0.3)'"
                     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 6px rgba(0, 0, 0, 0.3)'">
                    <div style='color: #22c55e; font-size: 2rem; margin-bottom: 0.5rem; font-weight: 600;'>HIGH</div>
                    <h3 style='color: #ffffff; margin-bottom: 1rem; font-size: 1.1rem;'>Intraday High</h3>
                    <div style='color: #ffffff; font-size: 2.2rem; font-weight: 600; margin-bottom: 0.75rem;'>
                        ${high_pred:,.2f}
                    </div>
                    <div style='background: linear-gradient(90deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.05) 100%);
                                padding: 0.5rem; border-radius: 6px; margin-bottom: 0.5rem;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='color: #8b92b0; font-size: 0.8rem;'>Confidence</span>
                            <span style='color: #22c55e; font-weight: 600; font-size: 0.9rem;'>{high_conf:.1f}%</span>
                        </div>
                    </div>
                    <div style='color: #8b92b0; font-size: 0.8rem; margin-bottom: 0.25rem;'>
                        <strong style='color: #ffffff;'>Range:</strong> ${high_lower:,.2f} - ${high_upper:,.2f}
                    </div>
                    <div style='color: #6b7280; font-size: 0.75rem;'>
                        Spread: ${high_range:,.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with pred_col3:
            close_conf = predictions['close']['confidence']
            close_pred = predictions['close']['prediction']
            close_lower = predictions['close']['lower_bound']
            close_upper = predictions['close']['upper_bound']
            close_range = close_upper - close_lower
            potential_gain = close_pred - last_close
            potential_gain_pct = (potential_gain / last_close) * 100
            
            st.markdown(f"""
                <div class='prediction-card' style='cursor: pointer; transition: all 0.3s ease;'
                     onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 16px rgba(168, 85, 247, 0.3)'"
                     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 6px rgba(0, 0, 0, 0.3)'">
                    <div style='color: #a855f7; font-size: 2rem; margin-bottom: 0.5rem; font-weight: 600;'>CLOSE</div>
                    <h3 style='color: #ffffff; margin-bottom: 1rem; font-size: 1.1rem;'>Closing Price</h3>
                    <div style='color: #ffffff; font-size: 2.2rem; font-weight: 600; margin-bottom: 0.75rem;'>
                        ${close_pred:,.2f}
                    </div>
                    <div style='background: linear-gradient(90deg, rgba(168, 85, 247, 0.2) 0%, rgba(168, 85, 247, 0.05) 100%);
                                padding: 0.5rem; border-radius: 6px; margin-bottom: 0.5rem;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='color: #8b92b0; font-size: 0.8rem;'>Confidence</span>
                            <span style='color: #a855f7; font-weight: 600; font-size: 0.9rem;'>{close_conf:.1f}%</span>
                        </div>
                    </div>
                    <div style='color: #8b92b0; font-size: 0.8rem; margin-bottom: 0.25rem;'>
                        <strong style='color: #ffffff;'>Range:</strong> ${close_lower:,.2f} - ${close_upper:,.2f}
                    </div>
                    <div style='color: {"#22c55e" if potential_gain >= 0 else "#ef4444"}; font-size: 0.75rem; font-weight: 600;'>
                        Expected: {potential_gain:+,.2f} ({potential_gain_pct:+.2f}%)
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Confidence gauge
        avg_confidence = (predictions['open']['confidence'] + 
                        predictions['high']['confidence'] + 
                        predictions['close']['confidence']) / 3
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        conf_col1, conf_col2 = st.columns([1, 2])
        
        with conf_col1:
            st.plotly_chart(create_prediction_gauge(avg_confidence), width="stretch")
        
        with conf_col2:
            st.markdown("""
                <div class='stats-card'>
                    <h3 style='color: #ffffff; margin-bottom: 1rem;'>Prediction Analysis</h3>
                </div>
            """, unsafe_allow_html=True)
            
            current_close = data['Close'].iloc[-1]
            predicted_close = predictions['close']['prediction']
            change = predicted_close - current_close
            change_pct = (change / current_close) * 100
            
            st.metric(
                label="Expected Change",
                value=format_price(change),
                delta=f"{change_pct:+.2f}%"
            )
            
            if avg_confidence >= 90:
                st.success("✅ **Very High Confidence** - Model is very certain about these predictions")
            elif avg_confidence >= 80:
                st.info("ℹ️ **High Confidence** - Model shows strong confidence in predictions")
            elif avg_confidence >= 70:
                st.warning("⚠️ **Moderate Confidence** - Predictions have reasonable confidence")
            else:
                st.error("❌ **Low Confidence** - Exercise caution with these predictions")
    
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
    
    # Additional Analytics Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <h2 style='color: #ffffff; margin-bottom: 1.5rem;'>Market Analysis & Technical Indicators</h2>
    """, unsafe_allow_html=True)
    
    anal_col1, anal_col2, anal_col3, anal_col4 = st.columns(4)
    
    # Calculate additional metrics
    returns = data['Close'].pct_change()
    avg_return = returns.mean() * 100
    max_gain = returns.max() * 100
    max_loss = returns.min() * 100
    win_rate = (returns > 0).sum() / len(returns) * 100
    
    with anal_col1:
        st.markdown(f"""
            <div class='stats-card' style='text-align: center; cursor: pointer; transition: all 0.3s ease;'
                 onmouseover="this.style.transform='scale(1.05)'"
                 onmouseout="this.style.transform='scale(1)'">
                <div style='color: #8b92b0; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;'>
                    Avg Daily Return
                </div>
                <div style='color: {"#22c55e" if avg_return >= 0 else "#ef4444"}; font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem;'>
                    {avg_return:+.3f}%
                </div>
                <div style='color: #6b7280; font-size: 0.7rem;'>
                    Over {len(data)} days
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with anal_col2:
        st.markdown(f"""
            <div class='stats-card' style='text-align: center; cursor: pointer; transition: all 0.3s ease;'
                 onmouseover="this.style.transform='scale(1.05)'"
                 onmouseout="this.style.transform='scale(1)'">
                <div style='color: #8b92b0; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;'>
                    Win Rate
                </div>
                <div style='color: #3b82f6; font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem;'>
                    {win_rate:.1f}%
                </div>
                <div style='color: #6b7280; font-size: 0.7rem;'>
                    Positive days
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with anal_col3:
        st.markdown(f"""
            <div class='stats-card' style='text-align: center; cursor: pointer; transition: all 0.3s ease;'
                 onmouseover="this.style.transform='scale(1.05)'"
                 onmouseout="this.style.transform='scale(1)'">
                <div style='color: #8b92b0; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;'>
                    Best Day
                </div>
                <div style='color: #22c55e; font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem;'>
                    +{max_gain:.2f}%
                </div>
                <div style='color: #6b7280; font-size: 0.7rem;'>
                    Maximum gain
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with anal_col4:
        st.markdown(f"""
            <div class='stats-card' style='text-align: center; cursor: pointer; transition: all 0.3s ease;'
                 onmouseover="this.style.transform='scale(1.05)'"
                 onmouseout="this.style.transform='scale(1)'">
                <div style='color: #8b92b0; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;'>
                    Worst Day
                </div>
                <div style='color: #ef4444; font-size: 2rem; font-weight: 600; margin-bottom: 0.5rem;'>
                    {max_loss:.2f}%
                </div>
                <div style='color: #6b7280; font-size: 0.7rem;'>
                    Maximum loss
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Volume Analysis
    st.markdown("<br>", unsafe_allow_html=True)
    vol_col1, vol_col2 = st.columns([2, 1])
    
    with vol_col1:
        st.markdown("""
            <h3 style='color: #ffffff; margin-bottom: 1rem;'>Volume Trends (Last 30 Days)</h3>
        """, unsafe_allow_html=True)
        
        # Create volume trend chart
        recent_data = data.tail(30)
        vol_fig = create_volume_chart(recent_data)
        st.plotly_chart(vol_fig, use_container_width=True)
    
    with vol_col2:
        st.markdown("""
            <h3 style='color: #ffffff; margin-bottom: 1rem;'>Volume Stats</h3>
        """, unsafe_allow_html=True)
        
        avg_vol_30d = data['Volume'].tail(30).mean()
        max_vol_30d = data['Volume'].tail(30).max()
        min_vol_30d = data['Volume'].tail(30).min()
        current_vol = data['Volume'].iloc[-1]
        
        st.markdown(f"""
            <div class='stats-card'>
                <div style='margin: 1rem 0;'>
                    <div style='color: #8b92b0; font-size: 0.75rem; margin-bottom: 0.25rem;'>Current Volume</div>
                    <div style='color: #ffffff; font-size: 1.5rem; font-weight: 600;'>{current_vol:,.0f}</div>
                </div>
                <div style='margin: 1rem 0;'>
                    <div style='color: #8b92b0; font-size: 0.75rem; margin-bottom: 0.25rem;'>30-Day Average</div>
                    <div style='color: #3b82f6; font-size: 1.2rem; font-weight: 600;'>{avg_vol_30d:,.0f}</div>
                </div>
                <div style='margin: 1rem 0;'>
                    <div style='color: #8b92b0; font-size: 0.75rem; margin-bottom: 0.25rem;'>30-Day High</div>
                    <div style='color: #22c55e; font-size: 1rem; font-weight: 600;'>{max_vol_30d:,.0f}</div>
                </div>
                <div style='margin: 1rem 0;'>
                    <div style='color: #8b92b0; font-size: 0.75rem; margin-bottom: 0.25rem;'>30-Day Low</div>
                    <div style='color: #ef4444; font-size: 1rem; font-weight: 600;'>{min_vol_30d:,.0f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if __name__ == "__main__":
    main()
