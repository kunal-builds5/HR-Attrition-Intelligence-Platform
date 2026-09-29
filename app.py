import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(
    page_title="HR Attrition Intelligence Platform",
    layout="wide"
)

# Custom CSS for UI Theme, Times New Roman Typography, and Clean Layout
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
    
    html, body, [class*="css"], .stMarkdown, h1, h2, h3, h4, h5, h6, p, label, button, input {
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    /* Clean Metric Cards Styling */
    [data-testid="stMetric"] {
        background-color: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
        font-size: 14px !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 26px !important;
    }

    /* Navigation Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161616;
        border-right: 1px solid #2d2d2d;
    }

    .sidebar-footer {
        position: fixed;
        bottom: 20px;
        font-family: 'Times New Roman', Times, serif;
        font-size: 16px;
        font-weight: bold;
        color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# Synthetic Data Generator
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Employee_ID': [f'EMP{i:04d}' for i in range(1, n + 1)],
        'Age': np.random.randint(22, 60, n),
        'Department': np.random.choice(['Sales', 'R&D', 'HR', 'Engineering', 'Marketing'], n),
        'Job_Role': np.random.choice(['Executive', 'Manager', 'Analyst', 'Specialist', 'Lead'], n),
        'Monthly_Income_INR': np.random.randint(30000, 250000, n),
        'Years_at_Company': np.random.randint(1, 20, n),
        'Work_Life_Balance_Score': np.random.randint(1, 5, n),
        'OverTime': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'Distance_From_Home_KM': np.random.randint(1, 50, n),
        'Performance_Rating': np.random.randint(1, 5, n),
    })
    
    churn_prob = (
        (df['OverTime'] == 'Yes') * 0.3 + 
        (df['Work_Life_Balance_Score'] < 2) * 0.3 + 
        (df['Distance_From_Home_KM'] > 30) * 0.2 + 
        (df['Years_at_Company'] < 3) * 0.2
    )
    df['Attrition'] = (churn_prob > 0.4).astype(int)
    return df

df = load_data()

# Clean Sidebar without emojis
st.sidebar.title("HR Intelligence OS")
st.sidebar.caption("Enterprise Attrition & Workforce Intelligence")

page = st.sidebar.radio(
    "Navigation Menu", 
    [
        "Executive Control Panel", 
        "Attrition Drivers & Behavioral Curves", 
        "Predictive Risk Simulator"
    ]
)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-footer">Kunal</div>', unsafe_allow_html=True)

# Plotly Base Chart Styling
chart_layout = dict(
    font=dict(family="Times New Roman", size=13, color="#e0e0e0"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=30, b=20)
)

# Page 1: Executive Control Panel
if page == "Executive Control Panel":
    st.title("HR Executive Control Panel")
    st.caption("Real-time organizational health metrics and attrition diagnostics.")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Headcount", f"{len(df):,}")
    col2.metric("Overall Attrition Rate", f"{(df['Attrition'].mean() * 100):.1f}%")
    col3.metric("Avg Monthly Salary", f"₹{df['Monthly_Income_INR'].mean():,.0f}")
    col4.metric("Avg Tenure", f"{df['Years_at_Company'].mean():.1f} Years")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Attrition Count by Department")
        fig_dept = px.histogram(
            df, x="Department", color="Attrition", barmode="group",
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        fig_dept.update_layout(**chart_layout)
        fig_dept.update_xaxes(showgrid=False)
        fig_dept.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
        st.plotly_chart(fig_dept, use_container_width=True)
        
    with col_right:
        st.subheader("Salary Distribution vs Attrition")
        fig_income = px.box(
            df, x="Department", y="Monthly_Income_INR", color="Attrition",
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        fig_income.update_layout(**chart_layout)
        fig_income.update_xaxes(showgrid=False)
        fig_income.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
        st.plotly_chart(fig_income, use_container_width=True)

# Page 2: Attrition Drivers
elif page == "Attrition Drivers & Behavioral Curves":
    st.title("Attrition Drivers & Behavioral Curves")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Overtime Impact on Employee Turnover")
        fig_overtime = px.histogram(
            df, x="OverTime", color="Attrition", barmode="group",
            color_discrete_map={0: '#3498db', 1: '#e74c3c'}
        )
        fig_overtime.update_layout(**chart_layout)
        fig_overtime.update_xaxes(showgrid=False)
        fig_overtime.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
        st.plotly_chart(fig_overtime, use_container_width=True)
        
    with col2:
        st.subheader("Work-Life Balance Score vs Attrition")
        fig_wlb = px.histogram(
            df, x="Work_Life_Balance_Score", color="Attrition", barmode="group",
            color_discrete_map={0: '#3498db', 1: '#e74c3c'}
        )
        fig_wlb.update_layout(**chart_layout)
        fig_wlb.update_xaxes(showgrid=False)
        fig_wlb.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333')
        st.plotly_chart(fig_wlb, use_container_width=True)

# Page 3: Risk Simulator
elif page == "Predictive Risk Simulator":
    st.title("Employee Flight Risk Predictor")
    st.caption("Adjust key employee attributes to calculate real-time attrition risk.")
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.slider("Employee Age", 20, 60, 30)
        income = st.slider("Monthly Income (INR)", 30000, 250000, 65000)
        years = st.slider("Years at Company", 1, 20, 3)
        distance = st.slider("Distance From Home (KM)", 1, 50, 15)
        
    with col_b:
        wlb = st.selectbox("Work Life Balance Score", [1, 2, 3, 4], index=1)
        overtime = st.selectbox("Does Overtime?", ["Yes", "No"])
        dept = st.selectbox("Department", ['Sales', 'R&D', 'HR', 'Engineering', 'Marketing'])
        
    if st.button("Calculate Attrition Probability"):
        risk_score = 0.15
        if overtime == "Yes": risk_score += 0.35
        if wlb <= 2: risk_score += 0.25
        if distance > 25: risk_score += 0.15
        if years < 2: risk_score += 0.10
        risk_score = min(risk_score, 0.95)
        
        st.markdown("---")
        if risk_score > 0.5:
            st.error(f"High Flight Risk Detected. Predicted Probability: {risk_score * 100:.1f}%")
            st.markdown("**Retention Recommendations:** Re-evaluate overtime requirements, offer flexible remote working days, or review compensation structure.")
        else:
            st.success(f"Low Flight Risk. Predicted Probability: {risk_score * 100:.1f}%")
