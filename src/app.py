import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="APL Logistics — Late Delivery Risk Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .main-header {
        background: linear-gradient(135deg, #1a1f2e, #2e3a4e);
        padding: 30px; border-radius: 12px;
        border-left: 5px solid #2E86AB;
        margin-bottom: 25px;
    }
    .metric-card {
        background: #1a1f2e; border-radius: 10px;
        padding: 20px; text-align: center;
        border: 1px solid #2e3a4e;
        margin-bottom: 15px;
    }
    .risk-high {
        background: linear-gradient(135deg, #3d1515, #5c1f1f);
        border: 2px solid #C73E1D; border-radius: 12px;
        padding: 25px; text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #3d2e10, #5c4520);
        border: 2px solid #F18F01; border-radius: 12px;
        padding: 25px; text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #0d2e2e, #1a4a4a);
        border: 2px solid #2E86AB; border-radius: 12px;
        padding: 25px; text-align: center;
    }
    .section-header {
        border-left: 4px solid #2E86AB;
        padding-left: 12px; margin: 20px 0 15px 0;
        font-size: 1.2em; font-weight: bold;
    }
    .info-box {
        background: #1a1f2e; border-radius: 8px;
        padding: 15px; border: 1px solid #2E86AB;
        margin: 10px 0;
    }
    .warning-box {
        background: #2e1f0d; border-radius: 8px;
        padding: 15px; border: 1px solid #F18F01;
        margin: 10px 0;
    }
    div[data-testid="stSidebar"] {
        background-color: #1a1f2e;
    }
    .stSelectbox label, .stNumberInput label,
    .stTextInput label { color: #a0aec0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS & ARTIFACTS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_DIR, 'models')
OUTPUTS_PATH = os.path.join(BASE_DIR, 'outputs')

@st.cache_resource
def load_artifacts():
    try:
        best_model           = joblib.load(os.path.join(MODELS_PATH, 'best_model.pkl'))
        encoders             = joblib.load(os.path.join(MODELS_PATH, 'encoders.pkl'))
        scaler_dict          = joblib.load(os.path.join(MODELS_PATH, 'scaler.pkl'))
        preprocessing_config = joblib.load(os.path.join(MODELS_PATH, 'preprocessing_config.pkl'))
        risk_thresholds      = joblib.load(os.path.join(MODELS_PATH, 'risk_thresholds.pkl'))
        return best_model, encoders, scaler_dict, preprocessing_config, risk_thresholds, None
    except Exception as e:
        return None, None, None, None, None, str(e)

best_model, encoders, scaler_dict, preprocessing_config, risk_thresholds, load_error = load_artifacts()

@st.cache_data
def load_outputs():
    try:
        scored   = pd.read_csv(os.path.join(OUTPUTS_PATH, 'scored_dataset.csv'))
        highrisk = pd.read_csv(os.path.join(OUTPUTS_PATH, 'high_risk_orders.csv'))
        drivers  = pd.read_csv(os.path.join(OUTPUTS_PATH, 'risk_drivers.csv'))
        return scored, highrisk, drivers, None
    except Exception as e:
        return None, None, None, str(e)

scored_df, high_risk_df, risk_drivers_df, output_error = load_outputs()

# ─────────────────────────────────────────────
# EXACT COLUMN ORDER FROM X_TRAIN
# ─────────────────────────────────────────────
COLUMN_ORDER = [
    'Type', 'Days for shipping (real)', 'Days for shipment (scheduled)',
    'Benefit per order', 'Sales per customer', 'Category Name',
    'Customer City', 'Customer Country', 'Customer Segment', 'Customer State',
    'Department Name', 'Market', 'Order City', 'Order Country',
    'Order Item Discount', 'Order Item Discount Rate', 'Order Item Product Price',
    'Order Item Profit Ratio', 'Order Item Quantity', 'Order Region',
    'Order State', 'Product Name', 'Shipping Mode',
    'shipping_delay_gap', 'shipping_pressure_index', 'is_express',
    'high_discount_flag', 'order_complexity_score', 'region_congestion_score'
]

# ─────────────────────────────────────────────
# DROPDOWN VALUES
# ─────────────────────────────────────────────
SHIPPING_MODES    = ['Standard Class', 'Second Class', 'First Class', 'Same Day']
PAYMENT_TYPES     = ['CASH', 'DEBIT', 'PAYMENT', 'TRANSFER']
MARKETS           = ['Africa', 'Europe', 'LATAM', 'Pacific Asia', 'USCA']
ORDER_REGIONS     = ['Canada', 'Caribbean', 'Central Africa', 'Central America',
                     'Central Asia', 'East Africa', 'East of USA', 'Eastern Asia',
                     'Eastern Europe', 'North Africa', 'Northern Europe', 'Oceania',
                     'South America', 'South Asia', 'South of  USA ', 'Southeast Asia',
                     'Southern Africa', 'Southern Europe', 'US Center ', 'West Africa',
                     'West Asia', 'West of USA ', 'Western Europe']
CUSTOMER_SEGMENTS = ['Consumer', 'Corporate', 'Home Office']
CATEGORY_NAMES    = ['Accessories', 'As Seen on  TV!', 'Baby ', 'Baseball & Softball',
                     'Basketball', 'Books ', 'Boxing & MMA', 'CDs ', 'Cameras ',
                     'Camping & Hiking', 'Cardio Equipment', "Children's Clothing",
                     'Cleats', 'Computers', 'Consumer Electronics', 'Crafts', 'DVDs',
                     'Electronics', 'Fishing', 'Fitness Accessories', 'Garden',
                     "Girls' Apparel", 'Golf Apparel', 'Golf Bags & Carts', 'Golf Balls',
                     'Golf Gloves', 'Golf Shoes', 'Health and Beauty', 'Hockey',
                     'Hunting & Shooting', 'Indoor/Outdoor Games', "Kids' Golf Clubs",
                     'Lacrosse', "Men's Clothing", "Men's Footwear", "Men's Golf Clubs",
                     'Music', 'Pet Supplies', 'Shop By Sport', 'Soccer', 'Sporting Goods',
                     'Strength Training', 'Tennis & Racquet', 'Toys', 'Trade-In',
                     'Video Games', 'Water Sports', "Women's Apparel", "Women's Clothing",
                     "Women's Golf Clubs"]
DEPARTMENT_NAMES  = ['Apparel', 'Book Shop', 'Discs Shop', 'Fan Shop', 'Fitness',
                     'Footwear', 'Golf', 'Health and Beauty ', 'Outdoors',
                     'Pet Shop', 'Technology']
ORDER_COUNTRIES   = ['Afganistán','Albania','Alemania','Angola','Arabia Saudí',
                     'Argelia','Argentina','Armenia','Australia','Austria',
                     'Azerbaiyán','Bangladés','Barbados','Baréin','Belice','Benín',
                     'Bielorrusia','Bolivia','Bosnia y Herzegovina','Botsuana',
                     'Brasil','Bulgaria','Burkina Faso','Burundi','Bután','Bélgica',
                     'Camboya','Camerún','Canada','Chad','Chile','China','Chipre',
                     'Colombia','Corea del Sur','Costa Rica','Costa de Marfil',
                     'Croacia','Cuba','Dinamarca','Ecuador','Egipto','El Salvador',
                     'Emiratos Árabes Unidos','Eritrea','Eslovaquia','Eslovenia',
                     'España','Estados Unidos','Estonia','Etiopía','Filipinas',
                     'Finlandia','Francia','Gabón','Georgia','Ghana','Grecia',
                     'Guadalupe','Guatemala','Guayana Francesa','Guinea',
                     'Guinea Ecuatorial','Guinea-Bissau','Guyana','Haití','Honduras',
                     'Hong Kong','Hungría','India','Indonesia','Irak','Irlanda','Irán',
                     'Israel','Italia','Jamaica','Japón','Jordania','Kazajistán',
                     'Kenia','Kirguistán','Kuwait','Laos','Lesoto','Liberia','Libia',
                     'Lituania','Luxemburgo','Líbano','Macedonia','Madagascar',
                     'Malasia','Mali','Marruecos','Martinica','Mauritania','Moldavia',
                     'Mongolia','Montenegro','Mozambique','Myanmar (Birmania)','México',
                     'Namibia','Nepal','Nicaragua','Nigeria','Noruega','Nueva Zelanda',
                     'Níger','Omán','Pakistán','Panamá','Papúa Nueva Guinea','Paraguay',
                     'Países Bajos','Perú','Polonia','Portugal','Qatar','Reino Unido',
                     'República Centroafricana','República Checa',
                     'República Democrática del Congo','República Dominicana',
                     'República de Gambia','República del Congo','Ruanda','Rumania',
                     'Rusia','Senegal','Serbia','Sierra Leona','Singapur','Siria',
                     'Somalia','Sri Lanka','Suazilandia','SudAfrica','Sudán',
                     'Sudán del Sur','Suecia','Suiza','Surinam','Sáhara Occidental',
                     'Tailandia','Taiwán','Tanzania','Tayikistán','Togo',
                     'Trinidad y Tobago','Turkmenistán','Turquía','Túnez','Ucrania',
                     'Uganda','Uruguay','Uzbekistán','Venezuela','Vietnam','Yemen',
                     'Yibuti','Zambia','Zimbabue']

# Region congestion scores from EDA
REGION_CONGESTION_MAP = {
    'Canada': 0.488, 'West Africa': 0.528, 'Caribbean': 0.531,
    'Southern Africa': 0.533, 'West of USA ': 0.540, 'Oceania': 0.540,
    'Northern Europe': 0.540, 'South America': 0.543, 'Eastern Asia': 0.543,
    'Southern Europe': 0.544, 'North Africa': 0.545, 'Central America': 0.548,
    'US Center ': 0.552, 'West Asia': 0.553, 'Central Asia': 0.553,
    'Southeast Asia': 0.555, 'East of USA': 0.557, 'Eastern Europe': 0.557,
    'South of  USA ': 0.558, 'Western Europe': 0.558, 'East Africa': 0.559,
    'South Asia': 0.563, 'Central Africa': 0.580
}

READABLE_FEATURES = {
    'shipping_delay_gap':            'Shipping Delay Gap',
    'Type':                          'Payment Type',
    'Days for shipment (scheduled)': 'Scheduled Shipping Days',
    'Customer Country':              'Customer Country',
    'Order Country':                 'Order Country',
    'Market':                        'Market',
    'region_congestion_score':       'Regional Congestion Score',
    'Order State':                   'Order State',
    'Customer City':                 'Customer City',
    'Order City':                    'Order City',
    'Customer State':                'Customer State',
    'Shipping Mode':                 'Shipping Mode',
    'Customer Segment':              'Customer Segment',
    'Order Region':                  'Order Region',
    'Days for shipping (real)':      'Actual Shipping Days',
    'is_express':                    'Express Shipping',
    'shipping_pressure_index':       'Shipping Pressure Index',
    'order_complexity_score':        'Order Complexity Score',
    'high_discount_flag':            'High Discount Flag',
    'Order Item Quantity':           'Order Quantity',
    'Order Item Product Price':      'Product Price',
    'Order Item Discount Rate':      'Discount Rate',
    'Order Item Discount':           'Discount Amount',
    'Order Item Profit Ratio':       'Profit Ratio',
    'Benefit per order':             'Benefit Per Order',
    'Sales per customer':            'Sales Per Customer',
    'Category Name':                 'Product Category',
    'Department Name':               'Department',
    'Product Name':                  'Product Name',
}

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_risk_category(prob):
    if prob >= 0.7:
        return 'High Risk'
    elif prob >= 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def encode_value(encoders, col, value):
    try:
        le = encoders[col]
        if value in le.classes_:
            return int(le.transform([value])[0])
        else:
            return -1
    except:
        return -1

def preprocess_input(user_input, encoders, scaler_dict, preprocessing_config):
    # Step 1 — Feature Engineering
    shipping_delay_gap     = user_input['Days for shipping (real)'] - \
                             user_input['Days for shipment (scheduled)']
    shipping_pressure_index= user_input['Days for shipment (scheduled)'] / \
                             (user_input['Order Item Quantity'] + 1)
    is_express             = 1 if user_input['Shipping Mode'] in \
                             ['First Class', 'Same Day'] else 0
    high_discount_flag     = 1 if user_input['Order Item Discount Rate'] > 0.06 else 0
    order_complexity_score = user_input['Order Item Quantity'] * \
                             user_input['Order Item Product Price']
    region_congestion_score= REGION_CONGESTION_MAP.get(
                             user_input['Order Region'], 0.545)

    # Step 2 — Build row with EXACT column order from X_train
    row = {
        'Type':                          user_input['Type'],
        'Days for shipping (real)':      user_input['Days for shipping (real)'],
        'Days for shipment (scheduled)': user_input['Days for shipment (scheduled)'],
        'Benefit per order':             user_input['Benefit per order'],
        'Sales per customer':            user_input['Sales per customer'],
        'Category Name':                 user_input['Category Name'],
        'Customer City':                 'Caguas',      # default — hidden
        'Customer Country':              'EE. UU.',     # default — hidden
        'Customer Segment':              user_input['Customer Segment'],
        'Customer State':                'PR',          # default — hidden
        'Department Name':               user_input['Department Name'],
        'Market':                        user_input['Market'],
        'Order City':                    user_input['Order City'],
        'Order Country':                 user_input['Order Country'],
        'Order Item Discount':           user_input['Order Item Discount'],
        'Order Item Discount Rate':      user_input['Order Item Discount Rate'],
        'Order Item Product Price':      user_input['Order Item Product Price'],
        'Order Item Profit Ratio':       user_input['Order Item Profit Ratio'],
        'Order Item Quantity':           user_input['Order Item Quantity'],
        'Order Region':                  user_input['Order Region'],
        'Order State':                   user_input['Order State'],
        'Product Name':                  'Accessories', # default — hidden
        'Shipping Mode':                 user_input['Shipping Mode'],
        'shipping_delay_gap':            shipping_delay_gap,
        'shipping_pressure_index':       shipping_pressure_index,
        'is_express':                    is_express,
        'high_discount_flag':            high_discount_flag,
        'order_complexity_score':        order_complexity_score,
        'region_congestion_score':       region_congestion_score,
    }

    # Step 3 — Create dataframe with exact column order
    df_input = pd.DataFrame([row])[COLUMN_ORDER]

    # Step 4 — Encode categorical columns
    categorical_cols = preprocessing_config['categorical_cols']
    for col in categorical_cols:
        if col in df_input.columns:
            df_input[col] = encode_value(encoders, col, str(df_input[col].iloc[0]))

    # Step 5 — Scale numerical columns
    numerical_cols = preprocessing_config['numerical_cols']
    scaler = scaler_dict['standard_scaler']
    df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])

    return df_input

def predict_risk(df_input):
    prob = best_model.predict_proba(df_input)[0][1]
    category = get_risk_category(prob)
    return prob, category

def get_top_drivers(n=5):
    feature_names  = COLUMN_ORDER
    importances    = best_model.feature_importances_
    driver_series  = pd.Series(importances, index=feature_names)
    top            = driver_series.sort_values(ascending=False).head(n)
    result = []
    for feat, score in top.items():
        result.append({
            'Feature':    READABLE_FEATURES.get(feat, feat),
            'Importance': round(score, 4)
        })
    return result

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:15px 0;'>
        <h2 style='color:#2E86AB; margin:0;'>🚚 APL Logistics</h2>
        <p style='color:#a0aec0; font-size:0.85em; margin:5px 0 0 0;'>
            Late Delivery Risk Intelligence
        </p>
    </div>
    <hr style='border-color:#2e3a4e;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Risk Predictor", "📊 Risk Dashboard", "⚠️ Operations Panel"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style='border-color:#2e3a4e;'>
    <div style='color:#a0aec0; font-size:0.78em; padding:10px 0;'>
        <b>Model:</b> XGBoost (Tuned)<br>
        <b>ROC-AUC:</b> 0.9972<br>
        <b>Accuracy:</b> 97.87%<br>
        <b>Recall:</b> 99.80%<br>
        <b>Client:</b> APL Logistics (KWE Group)<br>
        <b>Platform:</b> Unified Mentor
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class='main-header'>
        <h1 style='color:#ffffff; margin:0; font-size:2em;'>
            🚚 Late Delivery Risk Prediction System
        </h1>
        <h3 style='color:#2E86AB; margin:8px 0 5px 0;'>
            APL Logistics (KWE Group) — Global Supply Chain Intelligence
        </h3>
        <p style='color:#a0aec0; margin:0;'>
            A machine learning-powered tool for supply chain professionals
            to proactively identify and manage late delivery risk
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class='metric-card'>
            <h2 style='color:#2E86AB; margin:0;'>97.87%</h2>
            <p style='color:#a0aec0; margin:5px 0 0 0;'>Model Accuracy</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card'>
            <h2 style='color:#F18F01; margin:0;'>0.9972</h2>
            <p style='color:#a0aec0; margin:5px 0 0 0;'>ROC-AUC Score</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card'>
            <h2 style='color:#2E86AB; margin:0;'>180,517</h2>
            <p style='color:#a0aec0; margin:5px 0 0 0;'>Orders Analyzed</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class='metric-card'>
            <h2 style='color:#C73E1D; margin:0;'>99.80%</h2>
            <p style='color:#a0aec0; margin:5px 0 0 0;'>Recall Score</p>
        </div>""", unsafe_allow_html=True)

    # Who is this for
    st.markdown("<div class='section-header'>🎯 Who Is This Tool For?</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='warning-box'>
        <p style='color:#F18F01; font-weight:bold; margin:0 0 10px 0; font-size:1.1em;'>
            ⚠️ This is a Professional Supply Chain Intelligence Tool
        </p>
        <p style='color:#e2e8f0; line-height:1.8; margin:0;'>
            This system is designed exclusively for <b>supply chain managers, logistics
            analysts, warehouse operations teams, and dispatch coordinators</b> at
            APL Logistics (KWE Group). It is <b>not a consumer-facing application</b>.<br><br>
            The reason this tool requires professional access is that it depends on
            operational data — such as scheduled shipping days, actual shipping days,
            shipping mode decisions, profit ratios, and order financial metrics — that
            are only available within APL Logistics' internal order management systems.
            A regular customer placing an order would not have access to these values.<br><br>
            <b>Intended users include:</b>
        </p>
        <ul style='color:#e2e8f0; line-height:2; margin:10px 0 0 0;'>
            <li>Operations managers reviewing pre-dispatch order queues</li>
            <li>Logistics analysts assessing shipment risk before dispatch</li>
            <li>Warehouse teams prioritizing high-risk orders for expedited handling</li>
            <li>Customer service teams preparing proactive delay notifications</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # About the project
    st.markdown("<div class='section-header'>📌 About This Project</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <p style='color:#e2e8f0; line-height:1.8;'>
        In global logistics networks, late deliveries cause SLA breaches, financial
        penalties, and customer churn. The traditional approach of handling delays
        <b>after they happen</b> is reactive and costly — it leads to emergency
        rerouting, last-minute escalations, and damaged customer relationships.<br><br>
        This project delivers a <b style='color:#2E86AB;'>forward-looking predictive
        intelligence system</b> built for APL Logistics (KWE Group) that flags orders
        likely to be delayed <b>before they are dispatched</b>. By analyzing key order,
        shipping, and operational attributes, the system computes a
        <b style='color:#F18F01;'>Late Delivery Probability Score</b> for each order
        and classifies it into <b>Low</b>, <b>Medium</b>, or <b>High Risk</b> —
        enabling operations teams to take proactive action.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Problem vs Solution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>❌ The Problem</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <ul style='color:#e2e8f0; line-height:2;'>
                <li>Unpredictable shipment delays across global markets</li>
                <li>High cost of reactive last-minute interventions</li>
                <li>No system to prioritize high-risk orders before dispatch</li>
                <li>No quantitative risk scores for operational planning</li>
                <li>Limited visibility into what drives delays</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='section-header'>✅ The Solution</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <ul style='color:#e2e8f0; line-height:2;'>
                <li>ML model predicts delay risk before order is dispatched</li>
                <li>Probability score (0–100%) per order</li>
                <li>Risk category: Low / Medium / High</li>
                <li>Top risk drivers explained for each prediction</li>
                <li>Operations panel for immediate action queue</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    # Methodology
    st.markdown("<div class='section-header'>⚙️ Methodology</div>",
                unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class='info-box' style='text-align:center;'>
            <h3 style='color:#2E86AB;'>📦 Data</h3>
            <p style='color:#a0aec0;'>180,517 orders<br>28 features<br>
            DataCo Supply Chain</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='info-box' style='text-align:center;'>
            <h3 style='color:#F18F01;'>🔧 Processing</h3>
            <p style='color:#a0aec0;'>Feature engineering<br>Label encoding<br>
            Standard scaling</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='info-box' style='text-align:center;'>
            <h3 style='color:#A23B72;'>⚙️ Model</h3>
            <p style='color:#a0aec0;'>XGBoost Classifier<br>
            RandomizedSearchCV<br>5-Fold CV</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class='info-box' style='text-align:center;'>
            <h3 style='color:#C73E1D;'>🎯 Output</h3>
            <p style='color:#a0aec0;'>Risk Probability<br>Risk Category<br>
            Key Risk Drivers</p>
        </div>""", unsafe_allow_html=True)

    # How to use
    st.markdown("<div class='section-header'>📖 How To Use This Tool</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <ol style='color:#e2e8f0; line-height:2.2;'>
            <li>Navigate to <b style='color:#2E86AB;'>🔍 Risk Predictor</b>
                from the sidebar</li>
            <li>Enter the order details from your internal order management system</li>
            <li>Click <b>Predict Delivery Risk</b></li>
            <li>Review the risk probability, category, and top risk drivers</li>
            <li>Take action based on the recommendation shown</li>
            <li>Use <b style='color:#2E86AB;'>📊 Risk Dashboard</b>
                for a portfolio-level overview</li>
            <li>Use <b style='color:#C73E1D;'>⚠️ Operations Panel</b>
                for the high-risk order action queue</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 2 — RISK PREDICTOR
# ─────────────────────────────────────────────
elif page == "🔍 Risk Predictor":
    st.markdown("""
    <div class='main-header'>
        <h2 style='color:#ffffff; margin:0;'>🔍 Order Risk Predictor</h2>
        <p style='color:#a0aec0; margin:8px 0 0 0;'>
            Enter order details from your internal system to assess
            late delivery risk before dispatch.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if load_error:
        st.error(f"Model loading failed: {load_error}. Please check the models/ folder.")
        st.stop()

    with st.form("prediction_form"):

        # ── SHIPPING INFORMATION ──
        st.markdown("<div class='section-header'>📦 Shipping Information</div>",
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            shipping_mode = st.selectbox(
                "Shipping Mode",
                SHIPPING_MODES,
                help="Shipping method assigned by logistics team"
            )
        with col2:
            days_scheduled = st.number_input(
                "Scheduled Shipping Days (days promised to customer)",
                min_value=1, max_value=4, value=4, step=1,
                help="Number of days committed for delivery at order time"
            )
        with col3:
            days_real = st.number_input(
                "Actual Shipping Days (estimated by operations)",
                min_value=0, max_value=10, value=4, step=1,
                help="Estimated actual days based on current operational capacity"
            )

        # ── ORDER INFORMATION ──
        st.markdown("<div class='section-header'>🛒 Order Information</div>",
                    unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            quantity = st.number_input(
                "Order Item Quantity",
                min_value=1, max_value=100, value=1, step=1
            )
        with col2:
            product_price = st.number_input(
                "Product Price ($)",
                min_value=0.0, max_value=2000.0, value=50.0, step=0.01
            )
        with col3:
            discount_rate = st.number_input(
                "Discount Rate (0.0 to 1.0)",
                min_value=0.0, max_value=1.0, value=0.0, step=0.01,
                help="Discount rate applied at checkout (e.g. 0.05 = 5%)"
            )
        with col4:
            discount_amount = st.number_input(
                "Discount Amount ($)",
                min_value=0.0, max_value=500.0, value=0.0, step=0.01
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            payment_type = st.selectbox("Payment Type", PAYMENT_TYPES)
        with col2:
            profit_ratio = st.number_input(
                "Order Item Profit Ratio (from order system)",
                min_value=-1.0, max_value=1.0, value=0.3, step=0.01,
                help="Profit ratio from internal order management system"
            )
        with col3:
            benefit_per_order = st.number_input(
                "Benefit Per Order ($, from order system)",
                min_value=-500.0, max_value=500.0, value=50.0, step=0.01,
                help="Net benefit per order from internal financial records"
            )

        col1, col2 = st.columns(2)
        with col1:
            sales_per_customer = st.number_input(
                "Sales Per Customer ($, from CRM)",
                min_value=0.0, max_value=2000.0, value=100.0, step=0.01,
                help="Total sales value for this customer from CRM system"
            )
        with col2:
            category_name = st.selectbox("Product Category", CATEGORY_NAMES)

        # ── ORDER LOCATION ──
        st.markdown("<div class='section-header'>📍 Order Location</div>",
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            market = st.selectbox("Market / Global Region", MARKETS)
        with col2:
            order_region = st.selectbox("Order Region", ORDER_REGIONS)

        col1, col2 = st.columns(2)
        with col1:
            order_country = st.selectbox("Order Country", ORDER_COUNTRIES)
        with col2:
            order_state = st.text_input(
                "Order State",
                placeholder="e.g. California",
                help="State where the order is being delivered"
            )

        order_city = st.text_input(
            "Order City",
            placeholder="e.g. Los Angeles",
            help="City where the order is being delivered"
        )

        # ── CUSTOMER & PRODUCT ──
        st.markdown("<div class='section-header'>👤 Customer & Product</div>",
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            customer_segment = st.selectbox(
                "Customer Segment",
                CUSTOMER_SEGMENTS,
                help="Account classification from CRM"
            )
        with col2:
            department_name = st.selectbox("Department Name", DEPARTMENT_NAMES)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔍 Predict Delivery Risk",
            use_container_width=True
        )

    # ── PREDICTION ──
    if submitted:
        user_input = {
            'Type':                          payment_type,
            'Days for shipping (real)':      int(days_real),
            'Days for shipment (scheduled)': int(days_scheduled),
            'Benefit per order':             float(benefit_per_order),
            'Sales per customer':            float(sales_per_customer),
            'Category Name':                 category_name,
            'Customer Segment':              customer_segment,
            'Department Name':               department_name,
            'Market':                        market,
            'Order City':                    order_city.strip() if order_city.strip() != '' else 'Caguas',
            'Order Country':                 order_country,
            'Order Item Discount':           float(discount_amount),
            'Order Item Discount Rate':      float(discount_rate),
            'Order Item Product Price':      float(product_price),
            'Order Item Profit Ratio':       float(profit_ratio),
            'Order Item Quantity':           int(quantity),
            'Order Region':                  order_region,
            'Order State':                   order_state.strip() if order_state.strip() != '' else 'PR',
            'Shipping Mode':                 shipping_mode,
        }

        try:
            df_input = preprocess_input(
                user_input, encoders, scaler_dict, preprocessing_config
            )
            prob, category = predict_risk(df_input)
            top_drivers    = get_top_drivers()

            st.markdown("---")
            st.markdown("### 📊 Prediction Result")

            # ── Risk Card ──
            if category == 'High Risk':
                st.markdown(f"""
                <div class='risk-high'>
                    <h1 style='color:#C73E1D; margin:0; font-size:3em;'>
                        {prob*100:.1f}%
                    </h1>
                    <h3 style='color:#ffffff; margin:10px 0 5px 0;'>
                        🔴 HIGH RISK — Immediate Action Required
                    </h3>
                    <p style='color:#e2e8f0; margin:0;'>
                        This order has a high probability of late delivery.
                        Escalate to the logistics team before dispatch.
                    </p>
                </div>""", unsafe_allow_html=True)
            elif category == 'Medium Risk':
                st.markdown(f"""
                <div class='risk-medium'>
                    <h1 style='color:#F18F01; margin:0; font-size:3em;'>
                        {prob*100:.1f}%
                    </h1>
                    <h3 style='color:#ffffff; margin:10px 0 5px 0;'>
                        🟡 MEDIUM RISK — Monitor Closely
                    </h3>
                    <p style='color:#e2e8f0; margin:0;'>
                        This order has a moderate probability of late delivery.
                        Flag for monitoring post-dispatch.
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='risk-low'>
                    <h1 style='color:#2E86AB; margin:0; font-size:3em;'>
                        {prob*100:.1f}%
                    </h1>
                    <h3 style='color:#ffffff; margin:10px 0 5px 0;'>
                        🟢 LOW RISK — On Track
                    </h3>
                    <p style='color:#e2e8f0; margin:0;'>
                        This order has a low probability of late delivery.
                        Standard processing applies.
                    </p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Top Risk Drivers ──
            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.markdown(
                    "<div class='section-header'>🔑 Global Risk Drivers</div>",
                    unsafe_allow_html=True)
                st.markdown("""
                <div class='info-box'>
                    <p style='color:#a0aec0; margin:0 0 10px 0; font-size:0.85em;'>
                    Top features driving late delivery risk across all orders,
                    based on model feature importance.
                    </p>
                </div>""", unsafe_allow_html=True)
                for i, driver in enumerate(top_drivers):
                    bar_pct = min(int(driver['Importance'] * 120), 100)
                    st.markdown(f"""
                    <div style='margin:8px 0; padding:10px 15px;
                                background:#1a1f2e; border-radius:8px;
                                border-left:3px solid #2E86AB;'>
                        <div style='display:flex; justify-content:space-between;'>
                            <span style='color:#e2e8f0; font-weight:500;'>
                                {i+1}. {driver['Feature']}
                            </span>
                            <span style='color:#2E86AB; font-weight:bold;'>
                                {driver['Importance']:.4f}
                            </span>
                        </div>
                        <div style='background:#2e3a4e; border-radius:4px;
                                    height:6px; margin-top:8px;'>
                            <div style='background:#2E86AB; height:6px;
                                        border-radius:4px; width:{bar_pct}%;'>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            with col2:
                st.markdown(
                    "<div class='section-header'>💡 Recommended Action</div>",
                    unsafe_allow_html=True)
                if category == 'High Risk':
                    st.markdown("""
                    <div class='info-box' style='border-color:#C73E1D;'>
                        <ul style='color:#e2e8f0; line-height:2; margin:0;'>
                            <li>Escalate to logistics team immediately</li>
                            <li>Consider switching shipping mode to
                                <b>Standard Class</b></li>
                            <li>Notify customer proactively about
                                potential delay</li>
                            <li>Flag order for priority warehouse handling</li>
                            <li>Review regional congestion for this
                                delivery area</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)
                elif category == 'Medium Risk':
                    st.markdown("""
                    <div class='info-box' style='border-color:#F18F01;'>
                        <ul style='color:#e2e8f0; line-height:2; margin:0;'>
                            <li>Flag order for post-dispatch monitoring</li>
                            <li>Prepare contingency if delay is confirmed</li>
                            <li>Verify shipping mode selection is optimal</li>
                            <li>Check regional dispatch capacity</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='info-box' style='border-color:#2E86AB;'>
                        <ul style='color:#e2e8f0; line-height:2; margin:0;'>
                            <li>Order is on track for timely delivery</li>
                            <li>Standard processing and monitoring applies</li>
                            <li>No immediate intervention required</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please verify all input fields and try again.")

# ─────────────────────────────────────────────
# PAGE 3 — RISK DASHBOARD
# ─────────────────────────────────────────────
elif page == "📊 Risk Dashboard":
    st.markdown("""
    <div class='main-header'>
        <h2 style='color:#ffffff; margin:0;'>📊 Risk Dashboard</h2>
        <p style='color:#a0aec0; margin:8px 0 0 0;'>
            Portfolio-level overview of late delivery risk across all analyzed orders
        </p>
    </div>
    """, unsafe_allow_html=True)

    if output_error:
        st.error(f"Could not load output files: {output_error}")
        st.stop()

    total_orders  = len(scored_df)
    high_risk_cnt = len(scored_df[scored_df['Risk_Category'] == 'High Risk'])
    med_risk_cnt  = len(scored_df[scored_df['Risk_Category'] == 'Medium Risk'])
    low_risk_cnt  = len(scored_df[scored_df['Risk_Category'] == 'Low Risk'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#2E86AB; margin:0;'>{total_orders:,}</h2>
            <p style='color:#a0aec0; margin:5px 0 0 0;'>Total Orders Scored</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#C73E1D; margin:0;'>{high_risk_cnt:,}</h2>
            <p style='color:#a0aec0; margin:5px 0 0 0;'>High Risk Orders</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#F18F01; margin:0;'>{med_risk_cnt:,}</h2>
            <p style='color:#a0aec0; margin:5px 0 0 0;'>Medium Risk Orders</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#2E86AB; margin:0;'>{low_risk_cnt:,}</h2>
            <p style='color:#a0aec0; margin:5px 0 0 0;'>Low Risk Orders</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Risk Category Distribution</div>",
                    unsafe_allow_html=True)
        risk_counts = scored_df['Risk_Category'].value_counts().reset_index()
        risk_counts.columns = ['Risk Category', 'Count']
        color_map = {
            'High Risk': '#C73E1D',
            'Medium Risk': '#F18F01',
            'Low Risk': '#2E86AB'
        }
        fig = px.bar(
            risk_counts, x='Risk Category', y='Count',
            color='Risk Category',
            color_discrete_map=color_map,
            template='plotly_dark',
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            plot_bgcolor='#1a1f2e', paper_bgcolor='#1a1f2e',
            showlegend=False, height=350,
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Probability Distribution</div>",
                    unsafe_allow_html=True)
        fig2 = px.histogram(
            scored_df, x='Late_Delivery_Probability',
            nbins=30, template='plotly_dark',
            color_discrete_sequence=['#2E86AB']
        )
        fig2.add_vline(x=0.4, line_dash='dash', line_color='#F18F01',
                       annotation_text='Medium Risk (0.4)')
        fig2.add_vline(x=0.7, line_dash='dash', line_color='#C73E1D',
                       annotation_text='High Risk (0.7)')
        fig2.update_layout(
            plot_bgcolor='#1a1f2e', paper_bgcolor='#1a1f2e',
            height=350, font=dict(color='#e2e8f0'),
            xaxis_title='Late Delivery Probability',
            yaxis_title='Count'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Feature Importance
    st.markdown(
        "<div class='section-header'>Global Risk Drivers — Feature Importance</div>",
        unsafe_allow_html=True)
    if risk_drivers_df is not None:
        top_drivers_display = risk_drivers_df.head(10).copy()
        top_drivers_display['Feature'] = top_drivers_display['Feature'].map(
            lambda x: READABLE_FEATURES.get(x, x.replace('_', ' ').title())
        )
        fig3 = px.bar(
            top_drivers_display.sort_values('Importance Score'),
            x='Importance Score', y='Feature',
            orientation='h',
            template='plotly_dark',
            color_discrete_sequence=['#2E86AB'],
            text='Importance Score'
        )
        fig3.update_traces(
            texttemplate='%{text:.4f}', textposition='outside')
        fig3.update_layout(
            plot_bgcolor='#1a1f2e', paper_bgcolor='#1a1f2e',
            height=420, font=dict(color='#e2e8f0'),
            xaxis_title='Importance Score', yaxis_title=''
        )
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 4 — OPERATIONS PANEL
# ─────────────────────────────────────────────
elif page == "⚠️ Operations Panel":
    st.markdown("""
    <div class='main-header'>
        <h2 style='color:#ffffff; margin:0;'>⚠️ Operations Action Panel</h2>
        <p style='color:#a0aec0; margin:8px 0 0 0;'>
            High risk orders requiring immediate attention —
            sorted by risk probability descending
        </p>
    </div>
    """, unsafe_allow_html=True)

    if output_error:
        st.error(f"Could not load output files: {output_error}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#C73E1D; margin:0;'>{len(high_risk_df):,}</h2>
            <p style='color:#a0aec0; margin:5px 0 0 0;'>High Risk Orders</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        avg_prob = high_risk_df['Late_Delivery_Probability'].mean() * 100
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#F18F01; margin:0;'>{avg_prob:.1f}%</h2>
            <p style='color:#a0aec0; margin:5px 0 0 0;'>Avg Risk Probability</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        correct = (high_risk_df['Actual_Late_Delivery_Risk'] == 1).sum()
        accuracy = correct / len(high_risk_df) * 100
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#2E86AB; margin:0;'>{accuracy:.1f}%</h2>
            <p style='color:#a0aec0; margin:5px 0 0 0;'>
            Prediction Accuracy on High Risk</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Filters</div>",
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider(
            "Risk Probability Threshold",
            min_value=0.5, max_value=1.0,
            value=0.7, step=0.05,
            help="Show only orders above this probability threshold"
        )
    with col2:
        top_n = st.selectbox(
            "Show Top N Orders",
            [50, 100, 200, 500, 'All'], index=0
        )

    filtered_df = high_risk_df[
        high_risk_df['Late_Delivery_Probability'] >= threshold
    ].copy()

    if top_n != 'All':
        filtered_df = filtered_df.head(int(top_n))

    st.markdown(f"""
    <div class='info-box'>
        <p style='color:#e2e8f0; margin:0;'>
            Showing <b style='color:#C73E1D;'>{len(filtered_df):,}</b>
            high risk orders with probability ≥ <b>{threshold}</b>
        </p>
    </div>""", unsafe_allow_html=True)

    display_cols = ['Late_Delivery_Probability', 'Risk_Category',
                    'Actual_Late_Delivery_Risk']
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(
        filtered_df[available_cols].style.format(
            {'Late_Delivery_Probability': '{:.2%}'}
        ),
        use_container_width=True,
        height=400
    )

    st.markdown("<br>", unsafe_allow_html=True)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download High Risk Orders CSV",
        data=csv,
        file_name='high_risk_orders_filtered.csv',
        mime='text/csv',
        use_container_width=True
    )