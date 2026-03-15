import streamlit as st
# MUST be the first Streamlit command
st.set_page_config(
    page_title="Suicide Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
choice = st.sidebar.selectbox(
    "Choose a module",
    ["Home", "📈 Descriptive Statistics", "📊 Probability Distribution", "🔮 Prediction Models"]
)
# Home Page
if choice == "Home":
    st.title("📊 Suicide Data Analysis Dashboard")
    st.markdown("""
    Welcome to the *Suicide Data Analysis Web App*.  
    Use the sidebar on the left to navigate between different modules:
    - 📈 *Smart Descriptive Statistics*
    - 📊 *Probability Distribution & Tabular Representation*
    - 🔮 *Prediction Models*
    This tool is designed to help you explore, visualize, and predict based on suicide data using various statistical and machine learning techniques.
    """)
    st.markdown("---")
    st.markdown("👨‍💻 Developed by STATSQUAD · 🔍 Data Source: Kaggle")
# Module: Descriptive Stats
elif choice == "📈 Descriptive Statistics":
    from Pages import descriptive_analysis
    descriptive_analysis.run()
# Module: Probability Distribution
elif choice == "📊 Probability Distribution":
      from Pages import probability_distribution
      probability_distribution.run()
# Module: Prediction Models
elif choice == "🔮 Prediction Models":
     from Pages import prediction
     prediction.run()

