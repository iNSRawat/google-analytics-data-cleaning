"""
Streamlit App for Google Analytics Data Cleaning & Visualization
Deploy this app on Streamlit Cloud for free interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning_analysis import DataCleaningAnalysis

# Page configuration
st.set_page_config(
    page_title="Google Analytics Data Cleaning & Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📊 Google Analytics Data Cleaning & Visualization")
st.markdown("Professional data wrangling & business intelligence project")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "Data Overview", "Data Cleaning", "Visualizations", "Quality Report"]
)

if page == "Home":
    st.header("Project Overview")
    st.markdown("""
    This project demonstrates professional-grade data cleaning, transformation, and visualization 
    techniques using real Google Analytics data.
    
    ### Key Features:
    - ✅ Data cleaning pipeline
    - ✅ Duplicate removal
    - ✅ Missing value handling
    - ✅ Bot traffic detection
    - ✅ Interactive visualizations
    - ✅ Data quality reports
    
    ### Dataset:
    - **150,000+** user session events
    - **85 MB** raw data
    - **99%+** data accuracy
    """)

elif page == "Data Overview":
    st.header("📈 Data Overview")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Google Analytics CSV file",
        type=['csv'],
        help="Upload your google_analytics_export.csv file"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        
        st.subheader("Data Types")
        st.dataframe(pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        }))

elif page == "Data Cleaning":
    st.header("🧹 Data Cleaning Pipeline")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file for cleaning",
        type=['csv'],
        key="cleaning_upload"
    )
    
    if uploaded_file is not None:
        if st.button("Run Cleaning Pipeline"):
            with st.spinner("Cleaning data..."):
                # Save uploaded file temporarily
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Initialize cleaner
                    cleaner = DataCleaningAnalysis(data_path=tmp_path)
                    
                    # Run pipeline
                    cleaner.execute_pipeline()
                    
                    st.success("✅ Data cleaning completed!")
                    
                    # Show results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Records", f"{len(cleaner.df):,}")
                    with col2:
                        st.metric("Cleaned Records", f"{len(cleaner.cleaned_df):,}")
                    
                    # Download cleaned data
                    cleaned_csv = cleaner.cleaned_df.to_csv(index=False)
                    st.download_button(
                        label="Download Cleaned Data",
                        data=cleaned_csv,
                        file_name="cleaned_data.csv",
                        mime="text/csv"
                    )
                finally:
                    os.unlink(tmp_path)

elif page == "Visualizations":
    st.header("📊 Data Visualizations")
    
    uploaded_file = st.file_uploader(
        "Upload cleaned CSV file",
        type=['csv'],
        key="viz_upload"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            st.subheader("Distribution Analysis")
            
            selected_col = st.selectbox("Select column to visualize", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            df[selected_col].hist(bins=50, ax=ax, edgecolor='black')
            ax.set_title(f'Distribution of {selected_col}')
            ax.set_xlabel(selected_col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df[selected_col].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[selected_col].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df[selected_col].std():.2f}")
            with col4:
                st.metric("Min/Max", f"{df[selected_col].min():.2f} / {df[selected_col].max():.2f}")
        else:
            st.info("No numeric columns found for visualization")

elif page == "Quality Report":
    st.header("📋 Data Quality Report")
    
    uploaded_file = st.file_uploader(
        "Upload data quality report CSV",
        type=['csv'],
        key="quality_upload"
    )
    
    if uploaded_file is not None:
        quality_df = pd.read_csv(uploaded_file)
        
        st.dataframe(quality_df)
        
        # Summary metrics
        passed = len(quality_df[quality_df['status'] == 'PASS'])
        warnings = len(quality_df[quality_df['status'] == 'WARNING'])
        failed = len(quality_df[quality_df['status'] == 'FAIL'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Passed", passed)
        with col2:
            st.metric("⚠️ Warnings", warnings)
        with col3:
            st.metric("❌ Failed", failed)

# Footer
st.markdown("---")

# Donation Section
st.markdown("### 💰 You can help me by Donating")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        <a href="https://www.buymeacoffee.com/nsrawat" target="_blank">
            <img src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee" style="width: 200px;">
        </a>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <a href="https://paypal.me/nsrawat" target="_blank">
            <img src="https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white" alt="PayPal" style="width: 200px;">
        </a>
        """,
        unsafe_allow_html=True
    )

# Author Credit
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center;'>
        <p>Made with ❤️ by <a href="https://nsrawat.in" target="_blank" style="text-decoration: none; color: #1f77b4;">N S Rawat</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
