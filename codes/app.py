"""
Karnataka Crop Intelligence
Agricultural Data Warehouse & Data Mining Dashboard

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Karnataka Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# PREMIUM AGRICULTURAL UI
# ============================================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700;800&display=swap');


/* =========================================================================
   GLOBAL VARIABLES
   ========================================================================= */

:root {

    --forest: #163A24;
    --forest-2: #214D2F;

    --leaf: #3F7D3A;
    --leaf-light: #6D965A;

    --sage: #8FAF72;
    --light-sage: #DDE8D2;

    --cream: #F7F5EC;
    --warm-white: #FCFBF7;

    --wheat: #D6A84F;
    --earth: #8B6B45;

    --text: #203126;
    --muted: #66776A;

    --border: #DDE7D7;

    --white: #FFFFFF;
}


/* =========================================================================
   MAIN APPLICATION
   ========================================================================= */

.stApp {

    background:

        radial-gradient(
            circle at 8% 8%,
            rgba(143,175,114,0.16) 0,
            transparent 24%
        ),

        radial-gradient(
            circle at 92% 18%,
            rgba(214,168,79,0.10) 0,
            transparent 23%
        ),

        radial-gradient(
            circle at 50% 100%,
            rgba(63,125,58,0.06) 0,
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #F7F5EC 0%,
            #EEF3E8 50%,
            #F8F6EE 100%
        );

    color: var(--text);

    font-family:
        'DM Sans',
        sans-serif;

}


/* =========================================================================
   SUBTLE AGRICULTURAL CROP-ROW PATTERN
   ========================================================================= */

.stApp::before {

    content: "";

    position: fixed;

    inset: 0;

    pointer-events: none;

    opacity: 0.25;

    z-index: 0;

    background-image:

        repeating-linear-gradient(
            12deg,
            transparent 0px,
            transparent 48px,
            rgba(63,125,58,0.045) 49px,
            rgba(63,125,58,0.045) 51px
        );

}


/* =========================================================================
   MAIN CONTAINER
   ========================================================================= */

.main .block-container {

    padding-top: 2rem;

    padding-bottom: 4rem;

    max-width: 1450px;

}


/* =========================================================================
   SIDEBAR
   ========================================================================= */

[data-testid="stSidebar"] {

    background:

        radial-gradient(
            circle at 50% 8%,
            rgba(143,175,114,0.15),
            transparent 34%
        ),

        linear-gradient(
            180deg,
            #102C1B 0%,
            #163A24 45%,
            #1D4A2C 100%
        );

    border-right:
        1px solid rgba(255,255,255,0.10);

}


[data-testid="stSidebar"] * {

    color: #F5F8F1 !important;

}


[data-testid="stSidebar"] .stRadio label {

    color: #D6E4D1 !important;

}


[data-testid="stSidebar"] .stRadio > div {

    gap: 7px;

}


[data-testid="stSidebar"] .stRadio label {

    padding: 10px 12px;

    border-radius: 10px;

    transition:
        background 0.2s ease,
        transform 0.2s ease;

}


[data-testid="stSidebar"] .stRadio label:hover {

    background:
        rgba(255,255,255,0.08);

    transform:
        translateX(2px);

}


/* =========================================================================
   HERO
   ========================================================================= */

.hero {

    position: relative;

    overflow: hidden;

    background:

        radial-gradient(
            circle at 85% 18%,
            rgba(214,168,79,0.20),
            transparent 25%
        ),

        radial-gradient(
            circle at 72% 95%,
            rgba(143,175,114,0.18),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #102C1B 0%,
            #1E5130 55%,
            #397343 100%
        );

    padding: 48px 52px;

    border-radius: 28px;

    color: white;

    margin-bottom: 32px;

    box-shadow:
        0 18px 55px rgba(22,58,36,0.22);

    border:
        1px solid rgba(255,255,255,0.10);

}


/* Large decorative wheat */

.hero::after {

    content: "🌾";

    position: absolute;

    right: 45px;

    bottom: -25px;

    font-size: 150px;

    opacity: 0.09;

    transform: rotate(-8deg);

}


.hero h1 {

    font-family:
        'Playfair Display',
        serif;

    color: white;

    font-size: 45px;

    font-weight: 800;

    margin: 0 0 14px 0;

    letter-spacing: -1px;

}


.hero p {

    color: #DCEBDD;

    font-size: 17px;

    max-width: 820px;

    margin: 0;

    line-height: 1.75;

}


.badge {

    display: inline-block;

    padding: 7px 16px;

    border-radius: 30px;

    background:
        rgba(255,255,255,0.12);

    color: #E8F0D8;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 1.2px;

    margin-bottom: 18px;

    border:
        1px solid rgba(255,255,255,0.15);

}


/* =========================================================================
   KPI CARDS
   ========================================================================= */

.kpi-card {

    position: relative;

    overflow: hidden;

    background:
        rgba(255,255,255,0.90);

    backdrop-filter:
        blur(12px);

    padding: 24px 26px;

    border-radius: 20px;

    border:
        1px solid var(--border);

    box-shadow:
        0 8px 28px rgba(31,61,39,0.07);

    min-height: 130px;

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease,
        border-color 0.25s ease;

}


.kpi-card::after {

    content: "🌱";

    position: absolute;

    right: -8px;

    bottom: -18px;

    font-size: 65px;

    opacity: 0.07;

}


.kpi-card:hover {

    transform:
        translateY(-5px);

    box-shadow:
        0 14px 35px rgba(31,61,39,0.13);

    border-color:
        #A8C294;

}


.kpi-label {

    font-size: 11px;

    font-weight: 700;

    color: #71836E;

    text-transform: uppercase;

    letter-spacing: 1px;

    margin-bottom: 7px;

}


.kpi-value {

    font-size: 32px;

    font-weight: 800;

    color: var(--forest);

    line-height: 1.2;

}


.kpi-description {

    font-size: 12px;

    color: #829080;

    margin-top: 6px;

}


/* =========================================================================
   SECTION TITLES
   ========================================================================= */

.section-title {

    color: var(--forest);

    font-family:
        'Playfair Display',
        serif;

    font-size: 28px;

    font-weight: 700;

    margin-top: 38px;

    margin-bottom: 7px;

}


.section-description {

    color: var(--muted);

    font-size: 14px;

    margin-bottom: 20px;

    line-height: 1.7;

}


/* =========================================================================
   RECOMMENDATION CARD
   ========================================================================= */

.recommendation-card {

    position: relative;

    overflow: hidden;

    background:

        radial-gradient(
            circle at 15% 20%,
            rgba(143,175,114,0.20),
            transparent 30%
        ),

        radial-gradient(
            circle at 85% 80%,
            rgba(214,168,79,0.10),
            transparent 25%
        ),

        linear-gradient(
            135deg,
            #EDF5E7,
            #FFFFFF
        );

    border:
        2px solid #8FAF72;

    border-radius: 28px;

    padding: 42px 35px;

    margin-top: 24px;

    margin-bottom: 28px;

    text-align: center;

    box-shadow:
        0 12px 40px rgba(31,61,39,0.10);

}


.recommendation-card::after {

    content: "🌾";

    position: absolute;

    right: 25px;

    bottom: -25px;

    font-size: 110px;

    opacity: 0.08;

}


.recommendation-label {

    color: #638253;

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 2.5px;

    font-weight: 700;

}


.recommendation-crop {

    color: var(--forest);

    font-family:
        'Playfair Display',
        serif;

    font-size: 52px;

    font-weight: 800;

    margin: 10px 0;

}


.recommendation-confidence {

    color: #4E7047;

    font-size: 17px;

}


/* =========================================================================
   INFO CARDS
   ========================================================================= */

.info-card {

    background:
        rgba(255,255,255,0.90);

    backdrop-filter:
        blur(10px);

    border:
        1px solid var(--border);

    border-radius: 20px;

    padding: 26px;

    height: 100%;

    box-shadow:
        0 7px 25px rgba(31,61,39,0.05);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;

}


.info-card:hover {

    transform:
        translateY(-3px);

    box-shadow:
        0 12px 32px rgba(31,61,39,0.09);

}


.info-card h3 {

    color: var(--forest);

    margin-top: 0;

    margin-bottom: 12px;

    font-size: 18px;

}


.info-card p {

    color: #58705D;

    font-size: 14px;

    line-height: 1.75;

    margin-bottom: 0;

}


/* =========================================================================
   KDD PIPELINE
   ========================================================================= */

.pipeline {

    display: flex;

    align-items: center;

    justify-content: center;

    gap: 8px;

    flex-wrap: wrap;

    margin: 25px 0 35px 0;

    padding: 25px;

    background:
        rgba(255,255,255,0.88);

    backdrop-filter:
        blur(10px);

    border-radius: 22px;

    border:
        1px solid var(--border);

    box-shadow:
        0 7px 25px rgba(31,61,39,0.05);

}


.pipeline-step {

    position: relative;

    background:

        linear-gradient(
            135deg,
            #F0F5EA,
            #E5EFDD
        );

    border:
        1px solid #C8D9BD;

    border-radius: 15px;

    padding: 16px 22px;

    text-align: center;

    color: var(--forest);

    font-weight: 700;

    font-size: 13px;

    min-width: 105px;

    transition:
        all 0.25s ease;

}


.pipeline-step:hover {

    transform:
        translateY(-3px);

    background:
        #E1ECD8;

    border-color:
        #9AB47F;

}


.pipeline-arrow {

    color:
        #78956A;

    font-size: 21px;

    font-weight: 700;

}


/* =========================================================================
   BUTTONS
   ========================================================================= */

.stButton > button,
div[data-testid="stFormSubmitButton"] > button {

    background:

        linear-gradient(
            135deg,
            #285E36,
            #4A823D
        );

    color: white;

    border: none;

    border-radius: 13px;

    font-weight: 700;

    padding: 13px 28px;

    transition:
        all 0.25s ease;

}


.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {

    background:

        linear-gradient(
            135deg,
            #163A24,
            #285E36
        );

    color: white;

    transform:
        translateY(-2px);

    box-shadow:
        0 9px 25px rgba(22,58,36,0.25);

}


/* =========================================================================
   INPUTS
   ========================================================================= */

.stSelectbox > div > div,
.stMultiSelect > div > div {

    border-radius:
        12px !important;

    border-color:
        #D4DFCF !important;

    background:
        rgba(255,255,255,0.92) !important;

}


.stNumberInput input,
.stTextInput input {

    border-radius:
        12px !important;

}


.stSlider {

    padding-top:
        5px;

}


/* =========================================================================
   TABS
   ========================================================================= */

.stTabs [data-baseweb="tab-list"] {

    gap: 5px;

    background:
        #E5ECDD;

    border-radius: 14px;

    padding: 5px;

}


.stTabs [data-baseweb="tab"] {

    border-radius: 11px;

    padding: 9px 22px;

    font-weight: 600;

    color:
        #56705A;

}


/* =========================================================================
   EXPANDERS
   ========================================================================= */

[data-testid="stExpander"] {

    border:
        1px solid var(--border);

    border-radius:
        15px;

    overflow:
        hidden;

    background:
        rgba(255,255,255,0.70);

}


/* =========================================================================
   DATAFRAMES
   ========================================================================= */

[data-testid="stDataFrame"] {

    border-radius:
        15px;

    border:
        1px solid var(--border);

    overflow:
        hidden;

    box-shadow:
        0 5px 20px rgba(31,61,39,0.04);

}


/* =========================================================================
   ALERTS
   ========================================================================= */

[data-testid="stAlert"] {

    border-radius:
        15px;

    border:
        1px solid #D4E1CC;

}


/* =========================================================================
   DIVIDERS
   ========================================================================= */

hr {

    border:
        none;

    border-top:
        1px solid #DCE5D7;

    margin:
        30px 0;

}


/* =========================================================================
   FOOTER
   ========================================================================= */

.footer {

    text-align: center;

    color:
        #748373;

    font-size:
        12px;

    padding-top:
        35px;

    padding-bottom:
        25px;

    border-top:
        1px solid #DCE5D7;

    margin-top:
        50px;

}


.footer strong {

    color:
        #315C38;

}


/* =========================================================================
   SCROLLBAR
   ========================================================================= */

::-webkit-scrollbar {

    width:
        8px;

}


::-webkit-scrollbar-track {

    background:
        #EDF1E9;

}


::-webkit-scrollbar-thumb {

    background:
        #A5B99A;

    border-radius:
        10px;

}


::-webkit-scrollbar-thumb:hover {

    background:
        #718E66;

}


/* =========================================================================
   RESPONSIVE
   ========================================================================= */

@media (max-width: 900px) {

    .hero {

        padding:
            35px 28px;

    }

    .hero h1 {

        font-size:
            34px;

    }

    .recommendation-crop {

        font-size:
            40px;

    }

}

</style>
""")


# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "data/evaluation/rf_crop_model_tuned.pkl"
    )


# ============================================================================
# LOAD REFERENCE DATA
# ============================================================================

@st.cache_data
def load_reference_data():

    crop_mapping = pd.read_csv(
        "data/transformed/crop_label_mapping.csv"
    )

    scaler_params = pd.read_csv(
        "data/transformed/scaler_params.csv"
    )

    transformed_classification = pd.read_csv(
        "data/transformed/transformed_classification.csv",
        nrows=0
    )

    model_columns = [
        c
        for c in transformed_classification.columns
        if c != "target_crop_encoded"
    ]

    dim_district = pd.read_csv(
        "data/warehouse/Dim_District.csv"
    )

    district_clusters = pd.read_csv(
        "data/mining/district_clusters.csv"
    )

    silhouette = pd.read_csv(
        "data/evaluation/silhouette_scores.csv"
    )

    elbow = pd.read_csv(
        "data/mining/kmeans_elbow.csv"
    )

    trend = pd.read_csv(
        "data/transformed/transformed_trend.csv"
    )

    yearly_summary = pd.read_csv(
        "data/mining/yearly_trend_summary.csv"
    )

    crop_slopes = pd.read_csv(
        "data/mining/crop_trend_slopes.csv"
    )

    feature_importance = pd.read_csv(
        "data/mining/feature_importance.csv"
    )

    return {

        "crop_mapping":
            crop_mapping,

        "scaler_params":
            scaler_params,

        "model_columns":
            model_columns,

        "dim_district":
            dim_district,

        "district_clusters":
            district_clusters,

        "silhouette":
            silhouette,

        "elbow":
            elbow,

        "trend":
            trend,

        "yearly_summary":
            yearly_summary,

        "crop_slopes":
            crop_slopes,

        "feature_importance":
            feature_importance,
    }


# ============================================================================
# LOAD EVERYTHING
# ============================================================================

model = load_model()

ref = load_reference_data()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def metric_card(
    label,
    value,
    description=""
):

    st.html(f"""

    <div class="kpi-card">

        <div class="kpi-label">
            {label}
        </div>

        <div class="kpi-value">
            {value}
        </div>

        <div class="kpi-description">
            {description}
        </div>

    </div>

    """)


def section_header(
    title,
    description=""
):

    st.html(f"""

    <div class="section-title">
        {title}
    </div>

    <div class="section-description">
        {description}
    </div>

    """)


def build_input_vector(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    soil_type,
    season,
    district,
    region
):

    """
    Build one input row using exactly the same
    scaling and one-hot encoding used during training.
    """

    row = {
        col: 0
        for col in ref["model_columns"]
    }

    numeric_raw = {

        "N": N,

        "P": P,

        "K": K,

        "temperature":
            temperature,

        "humidity":
            humidity,

        "ph":
            ph,

        "rainfall":
            rainfall

    }

    scaler_params = ref["scaler_params"]

    scaler_classification = (
        scaler_params[
            scaler_params["dataset"]
            == "classification"
        ]
        .set_index("feature")
    )

    for feature, raw_value in numeric_raw.items():

        if feature in scaler_classification.index:

            mean = scaler_classification.loc[
                feature,
                "mean"
            ]

            std = scaler_classification.loc[
                feature,
                "std"
            ]

            if std == 0:

                std = 1

            scaled_column = (
                f"{feature}_scaled"
            )

            if scaled_column in row:

                row[scaled_column] = (
                    raw_value - mean
                ) / std

    categorical_values = [

        (
            "soil_type",
            soil_type
        ),

        (
            "season_name",
            season
        ),

        (
            "district_name",
            district
        ),

        (
            "region",
            region
        )

    ]

    for prefix, value in categorical_values:

        dummy_column = (
            f"{prefix}_{value}"
        )

        if dummy_column in row:

            row[dummy_column] = 1

    return pd.DataFrame(
        [row]
    )[ref["model_columns"]]


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.html("""

    <div style="
        text-align:center;
        padding:18px 0 12px 0;
    ">

        <div style="
            font-size:52px;
            line-height:1;
        ">
            🌾
        </div>

        <div style="
            font-size:20px;
            font-weight:700;
            margin-top:10px;
        ">
            Karnataka Crop
            <br>
            Intelligence
        </div>

        <div style="
            font-size:11px;
            opacity:0.70;
            margin-top:7px;
            letter-spacing:0.4px;
        ">
            AGRICULTURAL DATA
            <br>
            WAREHOUSE & MINING
        </div>

    </div>

    """)

    st.divider()

    st.html("""
    <div style="
        font-size:11px;
        letter-spacing:1px;
        color:#A8C294 !important;
        font-weight:700;
        margin-bottom:7px;
    ">
        EXPLORE
    </div>
    """)

    page = st.radio(

        "Navigation",

        [

            "🏠 Overview",

            "🌱 Crop Recommendation",

            "📍 Regional Analysis",

            "📈 Yield Trends"

        ],

        index=0,

        label_visibility="collapsed"

    )

    st.divider()

    st.html("""

    <div style="
        font-size:13px;
        line-height:1.7;
    ">

        <div style="
            color:#DDE8D2;
            font-weight:700;
            font-size:12px;
            letter-spacing:0.7px;
        ">
            DWDM MINI PROJECT
        </div>

        <div style="
            margin-top:6px;
            color:#C6D8C3;
        ">
            A KDD-based agricultural
            analytics system for
            Karnataka.
        </div>

        <br>

        <div style="
            color:#DDE8D2;
            font-weight:700;
            font-size:12px;
        ">
            KDD PIPELINE
        </div>

        <div style="
            margin-top:6px;
            color:#C6D8C3;
        ">
            Data → Warehouse
            → Mining
            → Evaluation
            → Knowledge
        </div>

    </div>

    """)

    st.divider()

    st.html("""

    <div style="
        font-size:12px;
        color:#A8C8A8;
    ">

        <span style="
            display:inline-block;
            width:8px;
            height:8px;
            background:#6FAF5F;
            border-radius:50%;
            margin-right:6px;
        "></span>

        System Online

        <span style="
            float:right;
            opacity:0.65;
        ">
            v2.0
        </span>

    </div>

    """)


# ============================================================================
# PAGE 1 — OVERVIEW
# ============================================================================

if page == "🏠 Overview":

    st.html("""

    <div class="hero">

        <div class="badge">
            🌾 AGRICULTURAL INTELLIGENCE • KARNATAKA
        </div>

        <h1>
            Karnataka Crop Intelligence
        </h1>

        <p>
            Turning agricultural data into actionable insights —
            from crop recommendation and district clustering
            to historical yield intelligence.
        </p>

    </div>

    """)

    # ------------------------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        metric_card(
            "DISTRICTS COVERED",
            ref["dim_district"].shape[0],
            "Regional coverage"
        )

    with col2:

        metric_card(
            "CROPS",
            ref["crop_mapping"].shape[0],
            "Crop classes"
        )

    with col3:

        metric_card(
            "RECOMMENDATION RECORDS",
            "21,960",
            "Training dataset"
        )

    with col4:

        metric_card(
            "YIELD DATA",
            "2004–2019",
            "Historical period"
        )

    # ------------------------------------------------------------------------
    # KDD PIPELINE
    # ------------------------------------------------------------------------

    section_header(
        "🌱 Knowledge Discovery Pipeline",
        "The complete workflow used to transform agricultural data into useful knowledge."
    )

    st.html("""

    <div class="pipeline">

        <div class="pipeline-step">
            📥<br>
            Data Collection
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            🎯<br>
            Data Selection
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            ⚙️<br>
            Transformation
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            🤖<br>
            Data Mining
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            📊<br>
            Evaluation
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            💡<br>
            Knowledge
        </div>

    </div>

    """)

    # ------------------------------------------------------------------------
    # MODEL PERFORMANCE
    # ------------------------------------------------------------------------

    section_header(
        "🤖 Machine Learning Performance",
        "Performance of the tuned Random Forest crop classification model."
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:

        metric_card(
            "TEST ACCURACY",
            "91.8%",
            "Random Forest"
        )

    with metric_col2:

        metric_card(
            "WEIGHTED F1",
            "0.917",
            "Classification quality"
        )

    with metric_col3:

        metric_card(
            "MACRO F1",
            "0.867",
            "Balanced performance"
        )

    with metric_col4:

        metric_card(
            "PRECISION",
            "0.891",
            "Overall precision"
        )

    # ------------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------------------

    section_header(
        "🌿 Feature Importance",
        "The features that contributed most strongly to the Random Forest crop classification model."
    )

    col1, col2 = st.columns(
        [1.5, 1]
    )

    with col1:

        importance = (
            ref["feature_importance"]
            .head(10)
            .copy()
        )

        importance = importance.sort_values(
            "importance",
            ascending=True
        )

        fig = px.bar(

            importance,

            x="importance",

            y="feature",

            orientation="h",

            text="importance",

            labels={
                "importance":
                    "Importance (%)",

                "feature":
                    "Feature"
            },

            color="importance",

            color_continuous_scale=[
                "#DDE8D2",
                "#8FAF72",
                "#3F7D3A",
                "#163A24"
            ]

        )

        fig.update_traces(

            texttemplate="%{x:.1f}%",

            textposition="outside",

            hovertemplate=
                "<b>%{y}</b><br>"
                "Importance: %{x:.2f}%"
                "<extra></extra>"

        )

        fig.update_layout(

            height=450,

            showlegend=False,

            coloraxis_showscale=False,

            margin=dict(
                l=10,
                r=50,
                t=20,
                b=10
            ),

            plot_bgcolor="rgba(0,0,0,0)",

            paper_bgcolor="rgba(0,0,0,0)"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.html("""

        <div class="info-card">

            <h3>
                🌱 Smart Crop Recommendation
            </h3>

            <p>
                A tuned Random Forest classifier
                learns relationships between soil,
                climate, season and district
                characteristics to recommend
                suitable crops.
            </p>

        </div>

        """)

        st.write("")

        st.html("""

        <div class="info-card">

            <h3>
                📍 Regional Intelligence
            </h3>

            <p>
                K-Means clustering identifies
                groups of districts with similar
                agricultural productivity profiles
                using yield, area, rainfall and
                yield-per-area characteristics.
            </p>

        </div>

        """)

        st.write("")

        st.html("""

        <div class="info-card">

            <h3>
                📈 Historical Trends
            </h3>

            <p>
                Historical agricultural records
                are analysed to understand crop
                yield patterns, cultivated area
                and long-term changes over time.
            </p>

        </div>

        """)

    # ------------------------------------------------------------------------
    # CLUSTER SUMMARY
    # ------------------------------------------------------------------------

    section_header(
        "📍 Regional Clustering",
        "K-Means was used to identify district productivity profiles."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        metric_card(
            "ELBOW METHOD",
            "k = 3",
            "Selected in mining phase"
        )

    with c2:

        best_silhouette = ref[
            "silhouette"
        ].loc[
            ref["silhouette"][
                "silhouette_score"
            ].idxmax()
        ]

        metric_card(
            "BEST SILHOUETTE",
            f"k = {int(best_silhouette['k'])}",
            f"Score: {best_silhouette['silhouette_score']:.3f}"
        )

    with c3:

        metric_card(
            "DISTRICTS",
            ref["dim_district"].shape[0],
            "Regional coverage"
        )

    st.info(
        "💡 The elbow method selected k=3, while silhouette analysis "
        "may favor a different value. Because the yield dataset contains "
        "a limited number of districts, these clustering results should "
        "be interpreted as exploratory rather than definitive regional classifications."
    )


# ============================================================================
# PAGE 2 — CROP RECOMMENDATION
# ============================================================================

elif page == "🌱 Crop Recommendation":

    st.html("""

    <div class="hero">

        <div class="badge">
            🌱 SMART AGRICULTURE • MACHINE LEARNING
        </div>

        <h1>
            Crop Recommendation
        </h1>

        <p>
            Discover which crop is most suitable for a given
            combination of soil, climate, season and district.
        </p>

    </div>

    """)

    district_options = sorted(

        ref["dim_district"][
            "district_name"
        ]
        .dropna()
        .unique()

    )

    district_to_region = dict(

        zip(

            ref["dim_district"][
                "district_name"
            ],

            ref["dim_district"][
                "region"
            ]

        )

    )

    # ------------------------------------------------------------------------
    # INPUT FORM
    # ------------------------------------------------------------------------

    section_header(
        "🧪 Enter Agricultural Conditions",
        "Provide the soil, climate and location conditions for the prediction."
    )

    with st.form(
        "recommendation_form"
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.html("""

            <div class="info-card">

                <h3>
                    🧪 Soil Conditions
                </h3>

                <p>
                    Specify the soil nutrients,
                    acidity and soil type.
                </p>

            </div>

            """)

            st.write("")

            N = st.slider(
                "Nitrogen (N)",
                0.0,
                300.0,
                90.0,
                step=5.0
            )

            P = st.slider(
                "Phosphorus (P)",
                0.0,
                150.0,
                40.0,
                step=5.0
            )

            K = st.slider(
                "Potassium (K)",
                0.0,
                220.0,
                40.0,
                step=5.0
            )

            ph = st.slider(
                "Soil pH",
                0.0,
                14.0,
                6.5,
                step=0.1
            )

            soil_type = st.selectbox(

                "Soil Type",

                [
                    "Alluvial",
                    "Black",
                    "Laterite",
                    "Loamy",
                    "Red",
                    "Sandy"
                ]

            )

        with col2:

            st.html("""

            <div class="info-card">

                <h3>
                    🌦️ Climate & Location
                </h3>

                <p>
                    Specify the environmental and
                    geographic conditions.
                </p>

            </div>

            """)

            st.write("")

            temperature = st.slider(
                "Temperature (°C)",
                0.0,
                50.0,
                25.0,
                step=0.5
            )

            humidity = st.slider(
                "Humidity (%)",
                0.0,
                100.0,
                70.0,
                step=5.0
            )

            rainfall = st.slider(
                "Rainfall (mm)",
                0.0,
                4000.0,
                200.0,
                step=50.0
            )

            season = st.selectbox(
                "Season",
                [
                    "Kharif",
                    "Rabi",
                    "Zaid"
                ]
            )

            district = st.selectbox(
                "District",
                district_options
            )

        st.write("")

        submitted = st.form_submit_button(
            "🌾 Generate Crop Recommendation",
            use_container_width=True
        )

    # ------------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------------

    if submitted:

        region = district_to_region[
            district
        ]

        X_input = build_input_vector(

            N=N,
            P=P,
            K=K,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
            soil_type=soil_type,
            season=season,
            district=district,
            region=region

        )

        try:

            probabilities = model.predict_proba(
                X_input
            )[0]

            crop_lookup = dict(

                zip(

                    ref["crop_mapping"][
                        "target_crop_encoded"
                    ],

                    ref["crop_mapping"][
                        "crop_name"
                    ]

                )

            )

            top_indices = np.argsort(
                probabilities
            )[::-1][:5]

            top_crop = crop_lookup.get(
                top_indices[0],
                "Unknown"
            )

            top_confidence = (
                probabilities[
                    top_indices[0]
                ] * 100
            )

            # ---------------------------------------------------------------
            # RESULT
            # ---------------------------------------------------------------

            st.html(f"""

            <div class="recommendation-card">

                <div class="recommendation-label">
                    🌟 Recommended Crop
                </div>

                <div class="recommendation-crop">
                    🌾 {top_crop}
                </div>

                <div class="recommendation-confidence">

                    Predicted probability:

                    <strong>
                        {top_confidence:.1f}%
                    </strong>

                </div>

            </div>

            """)

            # ---------------------------------------------------------------
            # INPUT SUMMARY
            # ---------------------------------------------------------------

            section_header(
                "🌱 Input Conditions",
                "Conditions provided to the machine-learning model."
            )

            s1, s2, s3, s4 = st.columns(4)

            with s1:

                metric_card(
                    "DISTRICT",
                    district,
                    region
                )

            with s2:

                metric_card(
                    "SOIL",
                    soil_type,
                    f"pH {ph:.1f}"
                )

            with s3:

                metric_card(
                    "SEASON",
                    season,
                    f"Rainfall {rainfall:.0f} mm"
                )

            with s4:

                metric_card(
                    "TEMPERATURE",
                    f"{temperature:.1f}°C",
                    f"Humidity {humidity:.0f}%"
                )

            # ---------------------------------------------------------------
            # TOP 5
            # ---------------------------------------------------------------

            section_header(
                "📊 Top 5 Candidate Crops",
                "The model's highest-probability crop classes for the entered conditions."
            )

            results = pd.DataFrame({

                "Crop": [

                    crop_lookup.get(
                        i,
                        "Unknown"
                    )

                    for i in top_indices

                ],

                "Probability": [

                    probabilities[i] * 100

                    for i in top_indices

                ]

            })

            fig = px.bar(

                results.sort_values(
                    "Probability"
                ),

                x="Probability",

                y="Crop",

                orientation="h",

                text="Probability",

                labels={

                    "Probability":
                        "Predicted probability (%)",

                    "Crop":
                        "Crop"

                },

                color="Probability",

                color_continuous_scale=[

                    "#DDE8D2",

                    "#8FAF72",

                    "#3F7D3A",

                    "#163A24"

                ]

            )

            fig.update_traces(

                texttemplate="%{text:.1f}%",

                textposition="outside",

                hovertemplate=
                    "<b>%{y}</b><br>"
                    "Probability: %{x:.1f}%"
                    "<extra></extra>"

            )

            fig.update_layout(

                height=400,

                coloraxis_showscale=False,

                margin=dict(
                    l=10,
                    r=50,
                    t=20,
                    b=10
                ),

                plot_bgcolor="rgba(0,0,0,0)",

                paper_bgcolor="rgba(0,0,0,0)"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            display_results = results.copy()

            display_results[
                "Probability"
            ] = display_results[
                "Probability"
            ].map(
                lambda x:
                    f"{x:.1f}%"
            )

            st.dataframe(

                display_results,

                hide_index=True,

                use_container_width=True

            )

            st.info(
                "ℹ️ The recommendation is a machine-learning "
                "prediction based on patterns learned from the "
                "project dataset. It should be treated as analytical "
                "decision support, not as a guarantee of crop performance."
            )

        except Exception as e:

            st.error(
                "Prediction could not be generated."
            )

            st.exception(e)


# ============================================================================
# PAGE 3 — REGIONAL ANALYSIS
# ============================================================================

elif page == "📍 Regional Analysis":

    st.html("""

    <div class="hero">

        <div class="badge">
            📍 AGRICULTURAL GEOGRAPHY • K-MEANS
        </div>

        <h1>
            Regional Productivity
        </h1>

        <p>
            Explore agricultural patterns across Karnataka
            through district-level clustering and productivity analysis.
        </p>

    </div>

    """)

    clusters = ref[
        "district_clusters"
    ].copy()

    # ------------------------------------------------------------------------
    # CLUSTER SUMMARY
    # ------------------------------------------------------------------------

    section_header(
        "🗺️ District Clusters",
        "K-Means groups districts using standardized productivity and environmental features."
    )

    cluster_ids = sorted(

        clusters[
            "cluster"
        ]
        .dropna()
        .unique()

    )

    summary_columns = st.columns(
        len(cluster_ids)
    )

    for column, cluster_id in zip(
        summary_columns,
        cluster_ids
    ):

        with column:

            count = (

                clusters[
                    "cluster"
                ]
                == cluster_id

            ).sum()

            metric_card(

                f"CLUSTER {int(cluster_id)}",

                count,

                "Districts"

            )

    # ------------------------------------------------------------------------
    # CLUSTER CHART
    # ------------------------------------------------------------------------

    st.write("")

    fig = px.bar(

        clusters.sort_values(
            "cluster"
        ),

        x="district_name",

        y="cluster",

        color="region",

        title="District Cluster Assignment",

        labels={

            "district_name":
                "District",

            "cluster":
                "Cluster",

            "region":
                "Region"

        }

    )

    fig.update_layout(

        height=470,

        margin=dict(
            l=10,
            r=10,
            t=60,
            b=90
        ),

        xaxis_tickangle=-45,

        plot_bgcolor="rgba(0,0,0,0)",

        paper_bgcolor="rgba(0,0,0,0)"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------------------------------
    # CLUSTER DETAILS
    # ------------------------------------------------------------------------

    section_header(
        "📋 Cluster Details",
        "Districts assigned to each productivity group."
    )

    for cluster_id in cluster_ids:

        cluster_data = clusters[
            clusters["cluster"]
            == cluster_id
        ]

        with st.expander(

            f"🌿 Cluster {int(cluster_id)} — "
            f"{len(cluster_data)} district(s)"

        ):

            columns_to_show = [

                "district_name",

                "region",

                "dominant_crop",

                "dominant_soil"

            ]

            available_columns = [

                c

                for c in columns_to_show

                if c in cluster_data.columns

            ]

            st.dataframe(

                cluster_data[
                    available_columns
                ],

                hide_index=True,

                use_container_width=True

            )

    st.divider()

    # ------------------------------------------------------------------------
    # ELBOW METHOD
    # ------------------------------------------------------------------------

    section_header(
        "📉 Elbow Method",
        "Inertia decreases as the number of clusters increases. The elbow method was used to select k=3."
    )

    fig_elbow = px.line(

        ref["elbow"],

        x="k",

        y="inertia",

        markers=True,

        title="K-Means Elbow Curve",

        labels={

            "k":
                "Number of clusters (k)",

            "inertia":
                "Inertia"

        }

    )

    fig_elbow.update_layout(

        height=400,

        plot_bgcolor="rgba(0,0,0,0)",

        paper_bgcolor="rgba(0,0,0,0)"

    )

    st.plotly_chart(

        fig_elbow,

        use_container_width=True

    )

    # ------------------------------------------------------------------------
    # SILHOUETTE
    # ------------------------------------------------------------------------

    section_header(
        "📐 Silhouette Analysis",
        "Silhouette score helps evaluate how well-separated the clusters are."
    )

    silhouette = ref[
        "silhouette"
    ].copy()

    fig_silhouette = px.line(

        silhouette,

        x="k",

        y="silhouette_score",

        markers=True,

        title="Silhouette Score by Number of Clusters",

        labels={

            "k":
                "Number of clusters (k)",

            "silhouette_score":
                "Silhouette score"

        }

    )

    fig_silhouette.update_layout(

        height=400,

        plot_bgcolor="rgba(0,0,0,0)",

        paper_bgcolor="rgba(0,0,0,0)"

    )

    st.plotly_chart(

        fig_silhouette,

        use_container_width=True

    )

    best_row = silhouette.loc[

        silhouette[
            "silhouette_score"
        ].idxmax()

    ]

    st.info(

        f"💡 The highest silhouette score in the evaluation "
        f"data occurs at k={int(best_row['k'])} "
        f"with a score of {best_row['silhouette_score']:.3f}. "
        f"The final clustering uses k=3 based on the elbow-method "
        f"decision from the mining phase."

    )


# ============================================================================
# PAGE 4 — YIELD TRENDS
# ============================================================================

elif page == "📈 Yield Trends":

    st.html("""

    <div class="hero">

        <div class="badge">
            📈 AGRICULTURAL HISTORY • 2004–2019
        </div>

        <h1>
            Yield Intelligence
        </h1>

        <p>
            Explore how crop yields, cultivated area and
            productivity patterns have changed across Karnataka.
        </p>

    </div>

    """)

    trend = ref[
        "trend"
    ].copy()

    # ------------------------------------------------------------------------
    # CROP SELECTION
    # ------------------------------------------------------------------------

    section_header(
        "🌱 Compare Crop Performance",
        "Select crops to compare their historical yield patterns."
    )

    crop_options = sorted(

        trend[
            "crop_name"
        ]
        .dropna()
        .unique()

    )

    selected_crops = st.multiselect(

        "Select crops",

        crop_options,

        default=crop_options[:4],

        label_visibility="collapsed"

    )

    # ------------------------------------------------------------------------
    # YIELD LINE CHART
    # ------------------------------------------------------------------------

    if selected_crops:

        filtered = trend[
            trend[
                "crop_name"
            ].isin(selected_crops)
        ]

        yearly_by_crop = (

            filtered

            .groupby(
                [
                    "year",
                    "crop_name"
                ],
                as_index=False
            )

            ["yield"]

            .mean()

        )

        fig = px.line(

            yearly_by_crop,

            x="year",

            y="yield",

            color="crop_name",

            markers=True,

            title="Average Yield by Year",

            labels={

                "yield":
                    "Average Yield",

                "year":
                    "Year",

                "crop_name":
                    "Crop"

            }

        )

        fig.update_layout(

            height=480,

            hovermode="x unified",

            plot_bgcolor="rgba(0,0,0,0)",

            paper_bgcolor="rgba(0,0,0,0)"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    else:

        st.warning(
            "Please select at least one crop."
        )

    st.divider()

    # ------------------------------------------------------------------------
    # TREND DIRECTION
    # ------------------------------------------------------------------------

    section_header(

        "📊 Trend Direction by Crop",

        "Linear regression slope indicates whether yield has generally increased or decreased over time."

    )

    slopes = (

        ref["crop_slopes"]

        .sort_values(
            "slope_per_year",
            ascending=False
        )

        .copy()

    )

    if "direction" in slopes.columns:

        slopes[
            "direction_display"
        ] = (

            slopes[
                "direction"
            ]

            .astype(str)

            .str.capitalize()

        )

    else:

        slopes[
            "direction_display"
        ] = np.where(

            slopes[
                "slope_per_year"
            ] >= 0,

            "Increasing",

            "Decreasing"

        )

    fig_slopes = px.bar(

        slopes,

        x="crop_name",

        y="slope_per_year",

        color="direction_display",

        title="Yield Trend Slope per Crop",

        labels={

            "crop_name":
                "Crop",

            "slope_per_year":
                "Slope per year",

            "direction_display":
                "Direction"

        }

    )

    fig_slopes.update_layout(

        height=500,

        xaxis_tickangle=-45,

        plot_bgcolor="rgba(0,0,0,0)",

        paper_bgcolor="rgba(0,0,0,0)"

    )

    st.plotly_chart(

        fig_slopes,

        use_container_width=True

    )

    st.caption(

        "Slope represents the average change in yield per year, "
        "estimated using linear regression on the crop's yearly "
        "average yield. Positive values indicate an increasing "
        "trend, while negative values indicate a decreasing trend."

    )

    # ------------------------------------------------------------------------
    # YEARLY SUMMARY
    # ------------------------------------------------------------------------

    st.divider()

    section_header(

        "📅 Yearly Agricultural Summary",

        "Overall yearly summary of yield, yield per area and cultivated area."

    )

    yearly = ref[
        "yearly_summary"
    ].copy()

    required_columns = [

        "year",

        "avg_yield",

        "avg_yield_per_area",

        "total_area",

        "record_count"

    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in yearly.columns

    ]

    if missing_columns:

        st.error(

            "The yearly summary file is missing the following "
            f"required columns: {missing_columns}"

        )

    else:

        col1, col2 = st.columns(2)

        # --------------------------------------------------------------------
        # AVERAGE YIELD
        # --------------------------------------------------------------------

        with col1:

            fig_yearly_yield = px.line(

                yearly,

                x="year",

                y="avg_yield",

                markers=True,

                title="Average Yield",

                labels={

                    "year":
                        "Year",

                    "avg_yield":
                        "Average Yield"

                }

            )

            fig_yearly_yield.update_layout(

                height=350,

                plot_bgcolor="rgba(0,0,0,0)",

                paper_bgcolor="rgba(0,0,0,0)"

            )

            st.plotly_chart(

                fig_yearly_yield,

                use_container_width=True

            )

        # --------------------------------------------------------------------
        # AVERAGE YIELD PER AREA
        # --------------------------------------------------------------------

        with col2:

            fig_yield_area = px.line(

                yearly,

                x="year",

                y="avg_yield_per_area",

                markers=True,

                title="Average Yield per Area",

                labels={

                    "year":
                        "Year",

                    "avg_yield_per_area":
                        "Average Yield per Area"

                }

            )

            fig_yield_area.update_layout(

                height=350,

                plot_bgcolor="rgba(0,0,0,0)",

                paper_bgcolor="rgba(0,0,0,0)"

            )

            st.plotly_chart(

                fig_yield_area,

                use_container_width=True

            )

        # --------------------------------------------------------------------
        # CULTIVATED AREA
        # --------------------------------------------------------------------

        st.subheader(
            "🌾 Total Cultivated Area"
        )

        fig_area = px.area(

            yearly,

            x="year",

            y="total_area",

            title="Total Cultivated Area by Year",

            labels={

                "year":
                    "Year",

                "total_area":
                    "Total Area"

            }

        )

        fig_area.update_layout(

            height=350,

            plot_bgcolor="rgba(0,0,0,0)",

            paper_bgcolor="rgba(0,0,0,0)"

        )

        st.plotly_chart(

            fig_area,

            use_container_width=True

        )

        # --------------------------------------------------------------------
        # RECORD COUNT
        # --------------------------------------------------------------------

        st.subheader(
            "📋 Number of Records by Year"
        )

        fig_records = px.bar(

            yearly,

            x="year",

            y="record_count",

            title="Agricultural Records by Year",

            labels={

                "year":
                    "Year",

                "record_count":
                    "Number of Records"

            },

            color="record_count",

            color_continuous_scale=[

                "#DDE8D2",

                "#8FAF72",

                "#3F7D3A",

                "#163A24"

            ]

        )

        fig_records.update_layout(

            height=350,

            coloraxis_showscale=False,

            plot_bgcolor="rgba(0,0,0,0)",

            paper_bgcolor="rgba(0,0,0,0)"

        )

        st.plotly_chart(

            fig_records,

            use_container_width=True

        )

        # --------------------------------------------------------------------
        # DATA TABLE
        # --------------------------------------------------------------------

        st.subheader(
            "📋 Yearly Summary Data"
        )

        st.dataframe(

            yearly,

            hide_index=True,

            use_container_width=True

        )


# ============================================================================
# FOOTER
# ============================================================================

st.html("""

<div class="footer">

    🌾

    <strong>
        Karnataka Crop Intelligence
    </strong>

    <br><br>

    Agricultural Data Warehouse & Data Mining

    <br>

    DWDM Mini Project • KDD Knowledge Presentation

    <br><br>

    <span style="
        opacity:0.65;
    ">
        Data → Warehouse → Mining → Evaluation → Knowledge
    </span>

</div>

""")