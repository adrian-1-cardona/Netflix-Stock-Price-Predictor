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

# Custom CSS - Modern Minimal Dark Theme (matching reference)
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background-color: #0b0b1e;
        padding: 0rem 1rem;
    }
    
    .stApp {
        background-color: #0b0b1e;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0b1e;
        border-right: 1px solid #1f1f33;
    }
    
    section[data-testid="stSidebar"] h1 {
        color: #ffffff;
        font-size: 1.25rem;
        font-weight: 600;
        padding: 0.5rem 0;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #fff;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #1a1a2e;
        color: #ffffff;
        border: 1px solid #2d2d44;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #232136;
        border-color: #3a3a54;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 1px solid #2d2d44;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8b92b0;
        border: none;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #ffffff !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    h1 {
        font-size: 1.4rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        font-size: 1.5rem !important;
        margin-top: 2rem !important;
    }
    
    h3 {
        font-size: 1.25rem !important;
        color: #3b82f6 !important;
    }
    
    /* Text */
    p, .stMarkdown {
        color: #8b92b0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
        border-bottom: 1px solid #1e2749;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8b92b0;
        border: none;
        padding: 1rem 0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #3b82f6;
        border-bottom: 2px solid #3b82f6;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 8px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        background-color: #1a1f3a;
        border: 1px solid #1e2749;
        border-radius: 8px;
        color: #ffffff;
    }
    
    /* Success */
    div[data-baseweb="notification"][kind="success"] {
        background-color: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
    }
    
    /* Info */
    div[data-baseweb="notification"][kind="info"] {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
    }
    
    /* Warning */
    div[data-baseweb="notification"][kind="warning"] {
        background-color: rgba(251, 191, 36, 0.1);
        border-left: 4px solid #fbbf24;
    }
    
    /* Error */
    div[data-baseweb="notification"][kind="error"] {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #1a1f3a;
        border: 1px solid #1e2749;
        border-radius: 8px;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #1a1f3a;
        border: 1px solid #1e2749;
        color: #ffffff;
    }
    
    /* Divider */
    hr {
        border-color: #1e2749;
        margin: 2rem 0;
    }
    
    /* Caption text */
    .caption, small {
        color: #6b7280 !important;
        font-size: 0.875rem;
    }
    
    /* Prediction Cards */
    .prediction-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #0f1629 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #1e2749;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    
    /* Stats Card */
    .stats-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #0f1629 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #1e2749;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Accent badges */
    .badge {
        background-color: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
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
        )
    )
    
    return fig
    
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
        xaxis_rangeslider_visible=False
    )
    
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
    # Sidebar
    with st.sidebar:
        st.markdown("""
<div style='padding: 0.5rem 0 2rem 0;'>
  <div style='font-size: 1.25rem; font-weight: bold; color: #fff; margin-bottom: 2.5rem; line-height: 1.3;'>Netflix Stock<br>Predictor</div>
  <div style='font-size: 0.95rem; color: #fff; margin-bottom: 1.2rem; cursor: pointer;'>Dashboard</div>
  <div style='font-size: 0.95rem; color: #fff; margin-bottom: 1.2rem; cursor: pointer;'>Update/Refresh Data</div>
  <div style='font-size: 0.95rem; color: #fff; margin-bottom: 1.2rem; cursor: pointer;'>Historical Data</div>
  <div style='font-size: 0.95rem; color: #fff; margin-bottom: 1.2rem; cursor: pointer;'>Analytics</div>
</div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Scenario selector (hidden label)
        scenario = st.selectbox("Market Scenario", ["Neutral", "Bullish", "Bearish"], key="scenario", label_visibility="collapsed")
        
        # Data refresh button
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Header with Export/Import buttons
    st.markdown("""
<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;'>
  <div style='font-size: 1.4rem; font-weight: 600; color: #fff;'>Report & Analytics</div>
</div>
    """, unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_spacer = st.columns([1, 1, 8])
    with col_btn1:
        st.button("Export", use_container_width=True)
    with col_btn2:
        st.button("Import", use_container_width=True)
    
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
    
    # Top metrics cards (interactive with hover effects)
    st.markdown(f"""
<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-bottom: 2rem; padding: 1.5rem; background: #1a1a2e; border: 1px solid #2d2d44; border-radius: 12px;'>
  <div style='cursor: pointer; transition: all 0.3s ease; padding: 0.5rem; border-radius: 8px;' 
       onmouseover="this.style.background='rgba(59, 130, 246, 0.1)'; this.style.transform='translateY(-2px)'"
       onmouseout="this.style.background='transparent'; this.style.transform='translateY(0)'">
    <div style='color: #8b92b0; font-size: 0.85rem; margin-bottom: 0.5rem;'>📊 Dashboard</div>
    <div style='color: #fff; font-size: 2rem; font-weight: 600; margin-bottom: 0.25rem;'>{len(data)}</div>
    <div style='color: #6b7280; font-size: 0.75rem;'>Last 7 day <span style='background: {'rgba(34, 197, 94, 0.2)' if change_7d >= 0 else 'rgba(239, 68, 68, 0.2)'}; color: {'#22c55e' if change_7d >= 0 else '#ef4444'}; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: 600;'>{change_7d:+.1f}%</span></div>
  </div>
  <div style='cursor: pointer; transition: all 0.3s ease; padding: 0.5rem; border-radius: 8px;'
       onmouseover="this.style.background='rgba(34, 197, 94, 0.1)'; this.style.transform='translateY(-2px)'"
       onmouseout="this.style.background='transparent'; this.style.transform='translateY(0)'">
    <div style='color: #8b92b0; font-size: 0.85rem; margin-bottom: 0.5rem;'>📈 Analytics</div>
    <div style='color: #fff; font-size: 2rem; font-weight: 600; margin-bottom: 0.25rem;'>{int(current_volume / 1000)}</div>
    <div style='color: #6b7280; font-size: 0.75rem;'>Last 7 day <span style='background: {'rgba(34, 197, 94, 0.2)' if volume_change >= 0 else 'rgba(239, 68, 68, 0.2)'}; color: {'#22c55e' if volume_change >= 0 else '#ef4444'}; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: 600;'>{volume_change:+.1f}%</span></div>
  </div>
  <div style='cursor: pointer; transition: all 0.3s ease; padding: 0.5rem; border-radius: 8px;'
       onmouseover="this.style.background='rgba(168, 85, 247, 0.1)'; this.style.transform='translateY(-2px)'"
       onmouseout="this.style.background='transparent'; this.style.transform='translateY(0)'">
    <div style='color: #8b92b0; font-size: 0.85rem; margin-bottom: 0.5rem;'>📅 Data Last Freshed</div>
    <div style='color: #fff; font-size: 2rem; font-weight: 600; margin-bottom: 0.25rem;'>{last_date.strftime('%b %d')}</div>
    <div style='color: #6b7280; font-size: 0.75rem;'>Last 7 day <span style='background: {'rgba(239, 68, 68, 0.2)' if days_old > 3 else 'rgba(34, 197, 94, 0.2)'}; color: {'#ef4444' if days_old > 3 else '#22c55e'}; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: 600;'>{days_old} days</span></div>
  </div>
  <div style='cursor: pointer; transition: all 0.3s ease; padding: 0.5rem; border-radius: 8px;'
       onmouseover="this.style.background='rgba(251, 191, 36, 0.1)'; this.style.transform='translateY(-2px)'"
       onmouseout="this.style.background='transparent'; this.style.transform='translateY(0)'">
    <div style='color: #8b92b0; font-size: 0.85rem; margin-bottom: 0.5rem;'>🎯 Model Accuracy</div>
    <div style='color: #fff; font-size: 2rem; font-weight: 600; margin-bottom: 0.25rem;'>{avg_confidence:.0f}%</div>
    <div style='color: #6b7280; font-size: 0.75rem;'>R² Score <span style='background: rgba(34, 197, 94, 0.2); color: #22c55e; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: 600;'>0.95</span></div>
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
<div style='padding: 1.5rem; background: #1a1a2e; border: 1px solid #2d2d44; border-radius: 12px; margin-bottom: 1.5rem;'>
  <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
    <div style='font-size: 1rem; font-weight: 600; color: #fff;'>Statics Overview</div>
    <div style='display: flex; gap: 1.5rem; align-items: center;'>
      <span style='color: #3b82f6; font-size: 0.85rem;'>● High</span>
      <span style='color: #fff; font-size: 0.85rem;'>● Close</span>
      <button style='background: #232136; color: #fff; border: 1px solid #444; border-radius: 6px; padding: 0.4rem 0.9rem; font-size: 0.85rem; cursor: pointer;'>6 Month</button>
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
<div style='padding: 1.5rem; background: #1a1a2e; border: 1px solid #2d2d44; border-radius: 12px;'>
  <div style='font-size: 1rem; font-weight: 600; color: #fff; margin-bottom: 1.5rem;'>Historical Data</div>
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
<div style='padding: 1.5rem; background: #1a1a2e; border: 1px solid #2d2d44; border-radius: 12px;'>
  <div style='color: #8b92b0; font-size: 0.85rem; margin-bottom: 0.5rem;'>💰 Total Market Cap</div>
  <div style='color: #fff; font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem;'>${market_cap / 1_000_000_000:.2f}B</div>
  <div style='color: #6b7280; font-size: 0.75rem; margin-bottom: 1rem;'><span style='background: {'rgba(34, 197, 94, 0.2)' if yoy_change >= 0 else 'rgba(239, 68, 68, 0.2)'}; color: {'#22c55e' if yoy_change >= 0 else '#ef4444'}; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: 600;'>{yoy_change:+.1f}%</span> YoY</div>
  
  <div style='width: 100%; height: 30px; display: flex; border-radius: 6px; overflow: hidden; margin-bottom: 1.5rem; cursor: pointer; transition: all 0.3s ease;'
       onmouseover="this.style.transform='scale(1.02)'"
       onmouseout="this.style.transform='scale(1)'">
    <div style='background: #3b82f6; width: 25%; transition: all 0.3s ease;' title='Current Price: ${last_close:,.2f}'></div>
    <div style='background: #22d3ee; width: 25%; transition: all 0.3s ease;' title='52W High: ${high_52w:,.2f}'></div>
    <div style='background: #a855f7; width: 15%; transition: all 0.3s ease;' title='Predictions: ${pred_close:,.2f}'></div>
    <div style='background: #f97316; width: 10%; transition: all 0.3s ease;' title='Volume: {avg_vol_30d / 1_000_000:.1f}M'></div>
    <div style='background: #22c55e; width: 25%; transition: all 0.3s ease;' title='Market Cap: ${market_cap / 1_000_000_000:.2f}B'></div>
  </div>
  
  <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;'>
    <div style='cursor: pointer; transition: all 0.3s ease; padding: 0.5rem; border-radius: 6px;'
         onmouseover="this.style.background='rgba(59, 130, 246, 0.1)'; this.style.transform='translateX(3px)'"
         onmouseout="this.style.background='transparent'; this.style.transform='translateX(0)'">
      <div style='color: #8b92b0; font-size: 0.75rem; margin-bottom: 0.25rem;'>● Current Price</div>
      <div style='color: #fff; font-size: 1rem; font-weight: 600;'>${last_close:,.2f}</div>
    </div>
    <div style='cursor: pointer; transition: all 0.3s ease; padding: 0.5rem; border-radius: 6px;'
         onmouseover="this.style.background='rgba(34, 211, 238, 0.1)'; this.style.transform='translateX(3px)'"
         onmouseout="this.style.background='transparent'; this.style.transform='translateX(0)'">
      <div style='color: #8b92b0; font-size: 0.75rem; margin-bottom: 0.25rem;'>● 52W High</div>
      <div style='color: #fff; font-size: 1rem; font-weight: 600;'>${high_52w:,.2f}</div>
    </div>
    <div style='cursor: pointer; transition: all 0.3s ease; padding: 0.5rem; border-radius: 6px;'
         onmouseover="this.style.background='rgba(168, 85, 247, 0.1)'; this.style.transform='translateX(3px)'"
         onmouseout="this.style.background='transparent'; this.style.transform='translateX(0)'">
      <div style='color: #8b92b0; font-size: 0.75rem; margin-bottom: 0.25rem;'>● Pred. Close</div>
      <div style='color: #fff; font-size: 1rem; font-weight: 600;'>${pred_close:,.2f}</div>
    </div>
    <div style='cursor: pointer; transition: all 0.3s ease; padding: 0.5rem; border-radius: 6px;'
         onmouseover="this.style.background='rgba(249, 115, 22, 0.1)'; this.style.transform='translateX(3px)'"
         onmouseout="this.style.background='transparent'; this.style.transform='translateX(0)'">
      <div style='color: #8b92b0; font-size: 0.75rem; margin-bottom: 0.25rem;'>● Avg Volume</div>
      <div style='color: #fff; font-size: 1rem; font-weight: 600;'>{avg_vol_30d / 1_000_000:.1f}M</div>
    </div>
    <div style='cursor: pointer; transition: all 0.3s ease; padding: 0.5rem; border-radius: 6px;'
         onmouseover="this.style.background='rgba(34, 197, 94, 0.1)'; this.style.transform='translateX(3px)'"
         onmouseout="this.style.background='transparent'; this.style.transform='translateX(0)'">
      <div style='color: #8b92b0; font-size: 0.75rem; margin-bottom: 0.25rem;'>● 52W Low</div>
      <div style='color: #fff; font-size: 1rem; font-weight: 600;'>${low_52w:,.2f}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    
    # Predictions Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <h2 style='color: #ffffff; margin-bottom: 1.5rem;'>Next Trading Day Predictions</h2>
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
