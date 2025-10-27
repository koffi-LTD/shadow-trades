import streamlit as st
import sys
import os
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(
    page_title="Shadow Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for gaming theme
st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --background-color: #1a1a2e;
            --secondary-bg: #16213e;
            --accent-color: #7b2cbf;
            --text-color: #e3e3e3;
            --success-color: #4CAF50;
            --warning-color: #ff9800;
            --error-color: #f44336;
        }
        
        /* Global styles */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        /* Headers */
        .main-header {
            font-size: 2.5rem;
            color: #9d4edd;
            margin-bottom: 1rem;
            text-shadow: 0 0 10px rgba(157, 78, 221, 0.5);
            font-weight: 600;
        }
        
        .sub-header {
            font-size: 1.5rem;
            color: #c77dff;
            margin-top: 2rem;
            text-shadow: 0 0 5px rgba(199, 125, 255, 0.3);
        }
        
        /* Cards */
        .metric-card {
            background-color: #16213e;
            border: 1px solid #7b2cbf;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.4);
        }
        
        /* Status colors */
        .highlight {
            color: #ff4b4b;
            font-weight: bold;
            text-shadow: 0 0 5px rgba(255, 75, 75, 0.3);
        }
        
        .success {
            color: #4ade80;
            font-weight: bold;
            text-shadow: 0 0 5px rgba(74, 222, 128, 0.3);
        }
        
        /* Info boxes */
        .info-box {
            background-color: #16213e;
            border-left: 4px solid #7b2cbf;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #16213e;
        }
        
        /* Buttons */
        .stButton button {
            background-color: #7b2cbf;
            color: white;
            border: none;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background-color: #9d4edd;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* Tables */
        .dataframe {
            background-color: #16213e;
            border: 1px solid #7b2cbf;
        }
        
        /* Links */
        a {
            color: #c77dff;
            text-decoration: none;
        }
        
        a:hover {
            color: #9d4edd;
            text-decoration: underline;
        }
        
        /* Disclaimer box */
        .disclaimer {
            background-color: #1a1a2e;
            border: 1px solid #7b2cbf;
            border-radius: 5px;
            padding: 1rem;
            margin-top: 2rem;
            color: #e3e3e3;
        }
        
        /* Metrics animation */
        @keyframes glow {
            0% { box-shadow: 0 0 5px rgba(123, 44, 191, 0.5); }
            50% { box-shadow: 0 0 20px rgba(123, 44, 191, 0.8); }
            100% { box-shadow: 0 0 5px rgba(123, 44, 191, 0.5); }
        }
        
        .glow-effect {
            animation: glow 2s infinite;
        }
    </style>
""", unsafe_allow_html=True)

# Header with styled title
st.markdown('<h1 class="main-header">MACD Trading Strategy Dashboard</h1>', unsafe_allow_html=True)

# Quick Stats Section
st.markdown('<h2 class="sub-header">Strategy Overview</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class="metric-card">
            <h3>Strategy Type</h3>
            <p>Momentum-based MACD</p>
        </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown("""
        <div class="metric-card">
            <h3>Timeframe</h3>
            <p>Daily</p>
        </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown("""
        <div class="metric-card">
            <h3>Risk Management</h3>
            <p>Pattern-based exits</p>
        </div>
    """, unsafe_allow_html=True)

# Strategy Rules Section
st.markdown('<h2 class="sub-header">Trading Rules</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="info-box">
            <h3>Entry Conditions</h3>
            <ul>
                <li><span class="success">Buy Signal:</span> 6 consecutive "lesser red" MACD histogram bars</li>
                <li>Each bar must be less negative than the previous</li>
                <li>Pattern indicates decreasing bearish momentum</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown("""
        <div class="info-box">
            <h3>Exit Conditions</h3>
            <ul>
                <li><span class="highlight">Sell Signal:</span> 5 consecutive "lesser green" MACD histogram bars</li>
                <li>20% minimum decrease between bars</li>
                <li><span class="highlight">Stop Loss:</span> 3 consecutive red candles with decreasing closes</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# MACD Parameters Section
st.markdown('<h2 class="sub-header">MACD Configuration</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class="metric-card">
            <h3>Fast Period</h3>
            <p>12 days</p>
        </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown("""
        <div class="metric-card">
            <h3>Slow Period</h3>
            <p>26 days</p>
        </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown("""
        <div class="metric-card">
            <h3>Signal Period</h3>
            <p>9 days</p>
        </div>
    """, unsafe_allow_html=True)

# Navigation Instructions
st.markdown('<h2 class="sub-header">Getting Started</h2>', unsafe_allow_html=True)
st.markdown("""
    <div class="info-box">
        <p>Use the sidebar navigation to access different features:</p>
        <ol>
            <li><strong>MACD Analysis:</strong> Real-time pattern detection and visualization</li>
            <li><strong>Strategy Backtesting:</strong> Test performance with historical data</li>
            <li><strong>Pattern Scanner:</strong> Find stocks currently forming MACD patterns</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
    <div class="disclaimer">
        <p><strong>Disclaimer:</strong> This trading strategy is for educational purposes only. 
        Always conduct your own research and consider your risk tolerance before making investment decisions.</p>
    </div>
""", unsafe_allow_html=True)