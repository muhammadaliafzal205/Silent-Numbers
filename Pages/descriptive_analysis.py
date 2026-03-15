import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    st.markdown("""
        <style>
        .main { background-color: #f4f6f9; }
        .css-1d391kg { background-color: #f4f6f9; }
        .reportview-container .markdown-text-container {
            font-family: 'Segoe UI', sans-serif;
            padding: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    st.title("🤖 Smart Descriptive Statistics & Visualization Tool")
    st.markdown("Upload your dataset and choose any column to visualize trends, stats, and confidence intervals.")
    uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    
        st.sidebar.title("🔧 Settings")
        selected_col = st.sidebar.selectbox("Select a Column", df.columns.tolist())
        tabs = st.tabs(["🗃 Original Data", "📊 Analysis", "📈 Visualization"])
        with tabs[0]:
            st.subheader("🗃 Original Dataset")
            st.dataframe(df)
        with tabs[1]:
            st.subheader(f"📊 Descriptive Statistics for: {selected_col}")
            data = df[selected_col].dropna()
            dtype = df[selected_col].dtype
            if np.issubdtype(dtype, np.number):
                mean = data.mean()
                median = data.median()
                mode_vals = data.mode()
                std_dev = data.std()
                range_val = data.max() - data.min()
                sem = std_dev / np.sqrt(len(data))
                t_score = stats.t.ppf(0.975, df=len(data) - 1)
                ci_lower = mean - t_score * sem
                ci_upper = mean + t_score * sem
                st.metric("Mean", f"{mean:.2f}")
                st.metric("Median", f"{median:.2f}")
                st.metric("Mode", f"{mode_vals.tolist()}")
                st.metric("Standard Deviation", f"{std_dev:.2f}")
                st.metric("Range", f"{range_val:.2f}")
                st.metric("95% Confidence Interval", f"({ci_lower:.2f}, {ci_upper:.2f})")
            else:
                mode_vals = data.mode()
                value_counts = data.value_counts()
    
                st.metric("Most Frequent Category (Mode)", f"{mode_vals.tolist()}")
                st.write("Category Counts:")
                st.dataframe(value_counts.reset_index().rename(columns={'index': 'Category', selected_col: 'Count'}))
        with tabs[2]:
            st.subheader(f"📈 Visualization for: {selected_col}")
            data = df[selected_col].dropna()
            dtype = df[selected_col].dtype
            if np.issubdtype(dtype, np.number):
                fig, ax = plt.subplots()
                sns.histplot(data, bins=20, kde=True, ax=ax, color='skyblue')
                ax.set_title(f"Distribution of {selected_col}")
                st.pyplot(fig)
            else:
                value_counts = data.value_counts()
                fig, ax = plt.subplots()
                sns.countplot(y=data, order=value_counts.index, palette="Set2", ax=ax)
                ax.set_title(f"Category Frequencies: {selected_col}")
                st.pyplot(fig)
    else:
        st.info("⬆ Please upload a CSV file to begin.")
