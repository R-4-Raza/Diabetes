import streamlit as st
import pandas as pd
import joblib

# ===== 1. Page Config =====
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# ===== Background Gradient CSS =====
page_bg = """
<style>
.stApp {
    background: linear-gradient(135deg, #f9f9d2, #fceabb, #ffd194);
    background-size: cover;
}

/* Sidebar styling - keep black but remove top strip */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2f2f2f, #4a4a4a, #5c5c5c);
    color: white;
    height: 100vh;
    margin-top: -60px;   /* removes top gap / strip */
    padding-top: 80px;   /* restore inside spacing */
    padding-left: 20px;
    padding-right: 20px;
}

/* Sidebar inner box */
.sidebar-box {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
    font-size: 15px;
    line-height: 1.6;
    color: white;
}

/* Go back button bottom-right */
div.go-back {
    position: fixed;
    bottom: 20px;
    right: 20px;
}

/* Style ONLY Submit and Predict buttons as green */
.stButton > button[kind="primary"] {
    background-color: #22c55e !important; /* Tailwind green-500 */
    color: white !important;
    border-radius: 8px;
    border: none;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ===== Sidebar =====
st.sidebar.markdown(
    """
    <div class="sidebar-box">
        <h3 style="color:#facc15;">🩺 Diabetes Prediction App</h3>
        <p><strong>Developed by:</strong> Ali Raza</p>
        <p><strong>Institute:</strong> University of Narowal</p>
        <p><strong>Email:</strong> ali@23razagmail.com</p>
        <hr style="margin: 10px 0; border: 1px solid #facc15;">
        <p>This app predicts the risk of diabetes based on user inputs.</p>
        <p><strong> Not a medical diagnosis.</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===== 2. Load Model =====
@st.cache_resource
def load_model():
    data = joblib.load("diabetes_model.pkl")  # Your saved model
    return data["pipeline"], data["features"], data["target"]

pipeline, features, target = load_model()

# ===== 3. Ask for User Name =====
if "username" not in st.session_state:
    st.session_state.username = None

if st.session_state.username is None:
    st.title("🩺 Diabetes Prediction App")
    name = st.text_input("👉 Please enter your name:")

    if st.button("Submit", type="primary"):   # ✅ green
        if name.strip() != "":
            st.session_state.username = name
            st.rerun()
        else:
            st.warning("⚠ Please enter a valid name before proceeding.")
else:
    # ===== Welcome Message =====
    st.title(f"Hi! {st.session_state.username}, Welcome to Diabetes Prediction App")
    st.write("Enter your health parameters below to check your diabetes risk.")

    # ===== 4. User Input Form =====
    st.subheader("Enter Your Details:")

    user_input = {}
    with st.form("input_form"):
        for feat in features:
            user_input[feat] = st.number_input(
                f"{feat}",
                min_value=0.0,
                max_value=500.0,
                value=25.0 if "BMI" in feat else 0.0,
                step=0.1
            )
        submit = st.form_submit_button("Predict")   # ✅ green

    # ===== 5. Prediction =====
    if submit:
        input_df = pd.DataFrame([user_input])
        prediction = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1] * 100

        st.subheader("📊 Prediction Results")
        st.write(f"**Diabetes Risk Probability:** `{prob:.2f}%`")

        if prediction == 1:
            st.error("⚠ The model predicts that you may have diabetes.")
            st.subheader("🩹 Precautions & Recommendations")
            st.markdown("""
            - Maintain a healthy diet low in sugar and refined carbs.  
            - Exercise regularly (at least 30 minutes/day).  
            - Monitor blood glucose levels regularly.  
            - Avoid smoking and excessive alcohol consumption.  
            - Visit your doctor for professional advice & treatment plans.
            """)
        else:
            st.success("✅ The model predicts that you are unlikely to have diabetes.")
            st.markdown("Keep following a healthy lifestyle to maintain your well-being.")

    # ===== 6. Go Back Button (Bottom-Right) =====
    with st.container():
        st.markdown('<div class="go-back">', unsafe_allow_html=True)
        if st.button("🔙 Go Back"):   # ⬅ stays default
            st.session_state.username = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ===== 7. Footer =====
st.markdown("---")
st.caption("Developed by Ali Raza using Streamlit & Scikit-learn")
