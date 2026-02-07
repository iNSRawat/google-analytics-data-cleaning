"""
Streamlit App for Google Analytics Data Cleaning & Visualization
Deploy this app on Streamlit Cloud for free interactive visualizations
"""

# #region agent log
import json, os
_log_debug_enabled = True
_log_path = None
try:
    _log_dir = os.path.join(os.path.dirname(__file__), '.cursor')
    _log_path = os.path.join(_log_dir, 'debug.log')
    os.makedirs(_log_dir, exist_ok=True)
except Exception:
    _log_debug_enabled = False
    _log_path = None

def log_debug(location, message, data=None, hypothesis_id=None):
    if not _log_debug_enabled:
        return
    try:
        with open(_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'location': location, 'message': message, 'data': data or {}, 'timestamp': __import__('time').time(), 'sessionId': 'debug-session', 'runId': 'run1', 'hypothesisId': hypothesis_id}) + '\n')
    except:
        pass
log_debug('streamlit_app.py:13', 'Starting imports', {}, 'A')
# #endregion

try:
    import streamlit as st
    log_debug('streamlit_app.py:17', 'streamlit imported successfully', {}, 'A')
except Exception as e:
    log_debug('streamlit_app.py:17', 'streamlit import failed', {'error': str(e)}, 'A')
    raise

# CRITICAL: st.set_page_config() MUST be called before any other Streamlit commands
# #region agent log
try:
    log_debug('streamlit_app.py:30', 'Before st.set_page_config (early)', {}, 'B')
    st.set_page_config(
        page_title="Google Analytics Data Cleaning & Visualization",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    log_debug('streamlit_app.py:37', 'st.set_page_config completed successfully (early)', {}, 'B')
except Exception as e:
    log_debug('streamlit_app.py:37', 'st.set_page_config failed (early)', {'error': str(e), 'error_type': type(e).__name__}, 'B')
    raise
# #endregion

try:
    import pandas as pd
    log_debug('streamlit_app.py:20', 'pandas imported successfully', {}, 'A')
except Exception as e:
    log_debug('streamlit_app.py:20', 'pandas import failed', {'error': str(e)}, 'A')
    raise

try:
    import numpy as np
    log_debug('streamlit_app.py:21', 'numpy imported successfully', {}, 'A')
except Exception as e:
    log_debug('streamlit_app.py:21', 'numpy import failed', {'error': str(e)}, 'A')
    raise

try:
    import matplotlib
    matplotlib.use('Agg')  # Set non-interactive backend for Streamlit Cloud
    import matplotlib.pyplot as plt
    log_debug('streamlit_app.py:25', 'matplotlib imported successfully with Agg backend', {}, 'A')
except Exception as e:
    log_debug('streamlit_app.py:25', 'matplotlib import failed', {'error': str(e)}, 'A')
    raise

try:
    import seaborn as sns
    log_debug('streamlit_app.py:23', 'seaborn imported successfully', {}, 'A')
except Exception as e:
    log_debug('streamlit_app.py:23', 'seaborn import failed', {'error': str(e)}, 'A')
    raise

try:
    from data_cleaning_analysis import DataCleaningAnalysis
    log_debug('streamlit_app.py:26', 'DataCleaningAnalysis imported successfully', {}, 'A')
except Exception as e:
    log_debug('streamlit_app.py:26', 'DataCleaningAnalysis import failed', {'error': str(e), 'error_type': type(e).__name__}, 'A')
    raise

# Footer function
def render_footer():
    """Render footer with donation section and author credit"""
    # #region agent log
    try:
        log_debug('streamlit_app.py:45', 'render_footer called', {}, 'E')
        # Create a container for the footer to ensure it always renders
        footer = st.container()
        log_debug('streamlit_app.py:48', 'footer container created', {}, 'E')
        with footer:
            # #endregion
            st.markdown("---")
            
            # Donation Section
            st.markdown("#### 💰 You can help me by Donating")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    '<a href="https://www.buymeacoffee.com/nsrawat?ref=Streamlit.app" target="_blank" style="text-decoration: none; color: white !important;"><div style="background-color: #FFDD00; color: white !important; border: none; padding: 12px 24px; border-radius: 5px; font-size: 16px; font-weight: bold; cursor: pointer; text-align: center; display: inline-block; width: 100%;">☕ Buy Me a Coffee</div></a>',
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    '<a href="https://paypal.me/NRawat710?ref=Streamlit.app" target="_blank" style="text-decoration: none; color: white !important;"><div style="background-color: #00457C; color: white !important; border: none; padding: 12px 24px; border-radius: 5px; font-size: 16px; font-weight: bold; cursor: pointer; text-align: center; display: inline-block; width: 100%;">💳 PayPal</div></a>',
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    '<a href="https://withupi.com/@nsrawat?ref=Streamlit.app" target="_blank" style="text-decoration: none; color: white !important;"><div style="background-color: #4CAF50; color: white !important; border: none; padding: 12px 24px; border-radius: 5px; font-size: 16px; font-weight: bold; cursor: pointer; text-align: center; display: inline-block; width: 100%;">₹ UPI</div></a>',
                    unsafe_allow_html=True
                )
            
            # Author Credit
            st.markdown("---")
            st.markdown(
                '<div style="text-align: center; padding: 20px 0;"><p style="margin: 0; font-size: 14px;">Made with ❤️ by <a href="https://nsrawat.in" target="_blank" style="text-decoration: underline; color: #1f77b4;">N S Rawat</a> | <a href="https://github.com/iNSRawat" target="_blank" style="text-decoration: underline; color: #1f77b4;">GitHub</a> | <a href="https://github.com/iNSRawat/data-cleaning-visualization" target="_blank" style="text-decoration: underline; color: #1f77b4;">Project Repo</a></p></div>',
                unsafe_allow_html=True
            )
            # #region agent log
            log_debug('streamlit_app.py:56', 'render_footer completed successfully', {}, 'E')
    except Exception as e:
        log_debug('streamlit_app.py:56', 'render_footer failed', {'error': str(e), 'error_type': type(e).__name__}, 'E')
        raise
    # #endregion

# Main app execution wrapped in error handler
# #region agent log
try:
    log_debug('streamlit_app.py:120', 'Starting main app execution', {}, 'F')
    # Title
    log_debug('streamlit_app.py:122', 'Before rendering title', {}, 'C')
    st.title("📊 Google Analytics Data Cleaning & Visualization")
    st.markdown("Professional data wrangling & business intelligence project")
    log_debug('streamlit_app.py:125', 'Title rendered successfully', {}, 'C')
except Exception as e:
    log_debug('streamlit_app.py:125', 'Title rendering failed', {'error': str(e), 'error_type': type(e).__name__}, 'C')
    st.error(f"Error initializing app: {str(e)}")
    st.exception(e)
    raise
# #endregion

# Sidebar
try:
    log_debug('streamlit_app.py:68', 'Before rendering sidebar', {}, 'C')
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Data Overview", "Data Cleaning", "Visualizations", "Quality Report"]
    )
    log_debug('streamlit_app.py:74', 'Sidebar rendered successfully', {'selected_page': page}, 'C')
except Exception as e:
    log_debug('streamlit_app.py:74', 'Sidebar rendering failed', {'error': str(e), 'error_type': type(e).__name__}, 'C')
    raise

if page == "Home":
    # #region agent log
    try:
        log_debug('streamlit_app.py:78', 'Rendering Home page', {}, 'D')
        st.header("Project Overview")
    except Exception as e:
        log_debug('streamlit_app.py:80', 'Home page rendering failed', {'error': str(e), 'error_type': type(e).__name__}, 'D')
        raise
    # #endregion
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
                    # #region agent log
                    log_debug('streamlit_app.py:142', 'Before initializing DataCleaningAnalysis', {'tmp_path': tmp_path}, 'A')
                    # Initialize cleaner
                    cleaner = DataCleaningAnalysis(data_path=tmp_path)
                    log_debug('streamlit_app.py:145', 'DataCleaningAnalysis initialized successfully', {}, 'A')
                    
                    # Run pipeline
                    log_debug('streamlit_app.py:148', 'Before execute_pipeline', {}, 'A')
                    cleaner.execute_pipeline()
                    log_debug('streamlit_app.py:150', 'execute_pipeline completed successfully', {}, 'A')
                    # #endregion
                    
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
            
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                df[selected_col].hist(bins=50, ax=ax, edgecolor='black')
                ax.set_title(f'Distribution of {selected_col}')
                ax.set_xlabel(selected_col)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
                plt.close(fig)  # Close figure to free memory
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                log_debug('streamlit_app.py:190', 'Visualization failed', {'error': str(e), 'column': selected_col}, 'D')
            
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

# Footer - Always visible at the bottom
render_footer()
