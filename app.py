
# ============================================================
# AUTOVALUE - USED CAR PRICE PREDICTION APPLICATION
# Built using Python, Streamlit and Machine Learning
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="AutoValue | Smart Car Valuation",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ------------------------------------------------------------
# 2. LOAD TRAINED MODEL AND CAR NAMES
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/car_price_model.pkl")


@st.cache_data
def load_car_names():
    return joblib.load("models/car_names.pkl")


model = load_model()
car_names = load_car_names()


# ------------------------------------------------------------
# 3. SESSION STATE
# ------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

if "latest" not in st.session_state:
    st.session_state.latest = None


def navigate(page):
    st.session_state.page = page


# ------------------------------------------------------------
# 4. CUSTOM CSS
# ------------------------------------------------------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: #080e19;
    color: #f4f7fb;
}

/* Remove default top spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* General text */
p, label, li {
    color: #b8c4d6;
}

/* Navigation buttons */
div.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 10px;
    border: 1px solid #263750;
    background: #111d30;
    color: #eaf2ff;
    font-weight: 600;
    transition: 0.25s ease;
}

div.stButton > button:hover {
    background: #1769d2;
    border-color: #3990ff;
    color: white;
}

/* Hero section */
.hero-tag {
    display: inline-block;
    color: #62aaff;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 700;
    padding: 10px 16px;
    border: 1px solid #244c7a;
    border-radius: 30px;
    background: rgba(30, 105, 190, 0.12);
    margin-bottom: 24px;
}

.hero-title {
    font-size: clamp(42px, 5vw, 70px);
    font-weight: 800;
    line-height: 1.12;
    color: #ffffff;
    letter-spacing: -2px;
    margin-bottom: 22px;
}

.hero-title span {
    color: #4a9cff;
}

.hero-description {
    color: #aab9cc;
    font-size: 17px;
    line-height: 1.8;
    max-width: 520px;
    margin-bottom: 30px;
}

/* Section headings */
.section-heading {
    color: #ffffff;
    font-size: 27px;
    font-weight: 750;
    margin-top: 35px;
    margin-bottom: 20px;
}

/* Feature cards */
.feature-card {
    background: linear-gradient(145deg, #14243a, #0d1727);
    border: 1px solid #253b57;
    border-radius: 17px;
    padding: 25px;
    min-height: 190px;
    transition: 0.3s ease;
}

.feature-card:hover {
    border-color: #3e91f5;
    transform: translateY(-4px);
}

.feature-icon {
    font-size: 30px;
    margin-bottom: 15px;
}

.feature-title {
    font-size: 19px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

.feature-description {
    color: #aab9cc;
    font-size: 14px;
    line-height: 1.7;
}

/* Metrics */
.metric-card {
    background: #111e31;
    border: 1px solid #253b57;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
}

.metric-label {
    color: #91a4bd;
    font-size: 13px;
    margin-bottom: 10px;
}

.metric-value {
    color: #60a9ff;
    font-size: 27px;
    font-weight: 800;
}

/* Prediction result */
.price-card {
    background: linear-gradient(135deg, #102c4e, #123c68);
    border: 1px solid #337fca;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 25px;
}

.price-label {
    color: #b8d7fa;
    font-size: 15px;
    letter-spacing: 1px;
}

.price-value {
    color: white;
    font-size: 43px;
    font-weight: 800;
    margin-top: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #71839c;
    border-top: 1px solid #203047;
    margin-top: 60px;
    padding: 25px;
    font-size: 13px;
}

/* Form containers */
div[data-testid="stForm"] {
    background: #101c2e;
    padding: 25px;
    border: 1px solid #263c58;
    border-radius: 16px;
}

/* Input fields */
.stSelectbox label,
.stNumberInput label {
    color: #dce8f7 !important;
    font-weight: 600;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# 5. TOP NAVIGATION BAR
# ------------------------------------------------------------

st.markdown("""
<div style="display:flex; align-items:center; gap:12px;
            margin-bottom:20px;">
    <span style="font-size:32px;">🚘</span>
    <span style="font-size:25px; font-weight:800; color:white;">
        Auto<span style="color:#4a9cff;">Value</span>
    </span>
    <span style="color:#71839c; font-size:12px;
                 margin-left:8px;">
        SMART CAR VALUATION
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='border-top:1px solid #263750;'></div>",
    unsafe_allow_html=True
)

nav1, nav2, nav3, nav4, nav5, nav6 = st.columns(
    [2.5, 1.1, 1.5, 1.1, 1.1, 0.9],
    gap="small"
)

with nav1:
    st.markdown(
        "<div style='padding-top:13px;color:#71839c;"
        "font-size:12px;'>EXPLORE AUTOVALUE</div>",
        unsafe_allow_html=True
    )

with nav2:
    if st.button("Home", key="nav_home"):
        navigate("Home")

with nav3:
    if st.button("Predict Price", key="nav_predict"):
        navigate("Predict Price")

with nav4:
    if st.button("Insights", key="nav_insights"):
        navigate("Insights")

with nav5:
    if st.button("History", key="nav_history"):
        navigate("History")

with nav6:
    if st.button("About", key="nav_about"):
        navigate("About")

st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 6. HOME PAGE
# ------------------------------------------------------------

if st.session_state.page == "Home":

    hero_text, hero_image = st.columns(
        [1.1, 1],
        gap="large"
    )

    with hero_text:

        # IMPORTANT: unsafe_allow_html=True renders the HTML
        # and applies the CSS classes defined above.

        st.markdown("""
        <div class="hero-tag">
            INTELLIGENT AUTOMOTIVE TECHNOLOGY
        </div>

        <div class="hero-title">
            Your car.<br>
            Its value.<br>
            <span>Reimagined.</span>
        </div>

        <div class="hero-description">
            Discover smarter used-car valuations with the power
            of machine learning. Make informed decisions before
            you buy or sell.
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Explore Price Predictor  →",
            key="hero_predict"
        ):
            navigate("Predict Price")
            st.rerun()

    with hero_image:

        # Car photograph displayed on the right side
        st.image(
            "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1100&q=85",
            use_container_width=True
        )

    # Feature section
    st.markdown(
        '<div class="section-heading">Why AutoValue?</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3, gap="large")

    with f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Machine Learning</div>
            <div class="feature-description">
                Powered by a trained Random Forest Regression
                model to estimate used car prices.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Instant Valuation</div>
            <div class="feature-description">
                Enter your car details and receive an estimated
                resale value within seconds.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Data Driven Insights</div>
            <div class="feature-description">
                Explore model performance, evaluation metrics
                and cross-validation results.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Project statistics
    st.markdown(
        '<div class="section-heading">Project at a Glance</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4, gap="medium")

    metrics = [
        ("Training Records", "3,577"),
        ("Test R² Score", "0.5602"),
        ("Average Test MAE", "₹1.33 L"),
        ("ML Algorithm", "Random Forest")
    ]

    for col, (label, value) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)


# ------------------------------------------------------------
# 7. PRICE PREDICTOR PAGE
# ------------------------------------------------------------

elif st.session_state.page == "Predict Price":

    st.markdown(
        '<div class="section-heading">Find Your Car\'s Value</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the details of your vehicle to get an estimated "
        "resale price using our machine learning model."
    )

    with st.form("prediction_form"):

        st.markdown("### 🚘 Vehicle Information")

        selected_car = st.selectbox(
            "Select Car Model",
            options=car_names
        )

        col1, col2 = st.columns(2)

        with col1:
            year = st.selectbox(
                "Manufacturing Year",
                options=list(range(2020, 1991, -1))
            )

            km_driven = st.number_input(
                "Kilometres Driven",
                min_value=0,
                max_value=1000000,
                value=30000,
                step=1000
            )

            fuel = st.selectbox(
                "Fuel Type",
                ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
            )

        with col2:
            seller_type = st.selectbox(
                "Seller Type",
                ["Individual", "Dealer", "Trustmark Dealer"]
            )

            transmission = st.selectbox(
                "Transmission",
                ["Manual", "Automatic"]
            )

            owner = st.selectbox(
                "Ownership",
                [
                    "First Owner",
                    "Second Owner",
                    "Third Owner",
                    "Fourth & Above Owner",
                    "Test Drive Car"
                ]
            )

        submitted = st.form_submit_button(
            "Calculate Estimated Price  →",
            use_container_width=True
        )

    if submitted:

        # Feature engineering must match the training notebook.
        brand = selected_car.split()[0]
        model_family = " ".join(selected_car.split()[:2])
        car_age = 2026 - year

        # Create input DataFrame with the exact training columns.
        input_data = pd.DataFrame([{
            "brand": brand,
            "model_family": model_family,
            "car_age": car_age,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner
        }])

        # Predict selling price
        predicted_price = float(model.predict(input_data)[0])

        # Avoid displaying negative prices
        predicted_price = max(0, predicted_price)

        # Save latest prediction
        st.session_state.latest = {
            "Car": selected_car,
            "Year": year,
            "Kilometres": km_driven,
            "Fuel": fuel,
            "Transmission": transmission,
            "Owner": owner,
            "Predicted Price": round(predicted_price, 2)
        }

        # Save prediction in current session history
        st.session_state.history.append(
            st.session_state.latest.copy()
        )

        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">
                ESTIMATED RESALE VALUE
            </div>
            <div class="price-value">
                ₹{predicted_price:,.0f}
            </div>
            <p style="color:#b8d7fa;">
                Your car's estimated market value
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Prediction Summary")

        st.write(f"**Car:** {selected_car}")
        st.write(f"**Manufacturing Year:** {year}")
        st.write(f"**Kilometres Driven:** {km_driven:,} km")
        st.write(f"**Fuel Type:** {fuel}")
        st.write(f"**Transmission:** {transmission}")
        st.write(f"**Ownership:** {owner}")

        st.info(
            "This is a machine learning estimate, not a guaranteed "
            "market price. Actual resale value may vary."
        )


# ------------------------------------------------------------
# 8. MODEL INSIGHTS PAGE
# ------------------------------------------------------------

elif st.session_state.page == "Insights":

    st.markdown(
        '<div class="section-heading">Model Performance & Insights</div>',
        unsafe_allow_html=True
    )

    st.write(
        "AutoValue uses a Random Forest Regression model "
        "trained on used-car sales data."
    )

    c1, c2, c3 = st.columns(3)

    insight_metrics = [
        ("R² Score", "0.5602"),
        ("Mean Absolute Error", "₹1,32,620"),
        ("Root Mean Squared Error", "₹3,76,395")
    ]

    for col, (label, value) in zip([c1, c2, c3], insight_metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Cross Validation Performance")

    cv_scores = [0.7767, 0.6276, 0.4692, 0.7647, 0.6485]

    cv_df = pd.DataFrame({
        "Fold": [
            "Fold 1",
            "Fold 2",
            "Fold 3",
            "Fold 4",
            "Fold 5"
        ],
        "R² Score": cv_scores
    })

    st.bar_chart(
        cv_df.set_index("Fold"),
        use_container_width=True
    )

    st.write("**Mean CV R² Score:** 0.6573")
    st.write("**Standard Deviation:** 0.1115")

    st.markdown("### How the Model Works")

    st.markdown("""
    **1. Data Preprocessing**

    Categorical features are converted into numerical
    representations using One-Hot Encoding.

    **2. Feature Engineering**

    Brand, model family and car age are derived from
    the original dataset.

    **3. Random Forest Regression**

    Multiple decision trees work together to estimate
    the selling price of a vehicle.

    **4. Log Transformation**

    The target price is transformed using `log1p`
    during training and converted back using `expm1`.
    """)

    st.warning(
        "Model predictions may be less reliable for rare car models "
        "and high-value premium vehicles because the training dataset "
        "contains fewer examples of these vehicles."
    )

    st.caption(
        "R² measures the proportion of variation in prices explained "
        "by the model. It should not be interpreted as prediction accuracy."
    )


# ------------------------------------------------------------
# 9. PREDICTION HISTORY PAGE
# ------------------------------------------------------------

elif st.session_state.page == "History":

    st.markdown(
        '<div class="section-heading">Your Prediction History</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear Prediction History"):
            st.session_state.history = []
            st.session_state.latest = None
            st.rerun()

    else:
        st.info(
            "No predictions yet. Visit the Price Predictor "
            "to calculate your first car valuation."
        )


# ------------------------------------------------------------
# 10. ABOUT PAGE
# ------------------------------------------------------------

elif st.session_state.page == "About":

    st.markdown(
        '<div class="section-heading">About AutoValue</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    AutoValue is a machine learning based used-car price
    prediction application developed to estimate the resale
    value of pre-owned vehicles.

    ### Project Objectives

    - Estimate used-car selling prices using machine learning.
    - Understand the relationship between vehicle features
      and resale value.
    - Provide an easy-to-use interface for price estimation.
    - Demonstrate the practical application of regression models.

    ### Technology Stack

    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Joblib
    - Random Forest Regression

    ### Dataset

    The project uses the Car Details from Car Dekho dataset.
    The data includes vehicle model, manufacturing year,
    kilometres driven, fuel type, seller type, transmission,
    ownership and selling price.

    ### Disclaimer

    AutoValue provides an estimated price based on historical
    dataset patterns. It does not guarantee the actual resale
    value of any vehicle.
    """)


# ------------------------------------------------------------
# 11. FOOTER
# ------------------------------------------------------------

st.markdown("""
<div class="footer">
    <div style="font-size:18px;font-weight:700;color:#dce8f7;">
        Auto<span style="color:#4a9cff;">Value</span>
    </div>
    Intelligent Automotive Technology | Machine Learning Project
    <br><br>
    Built with Python, Scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)
