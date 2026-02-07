"""
Gradio App for Google Analytics Data Cleaning & Visualization
Mirrors the functionality of streamlit_app.py
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

from data_cleaning_analysis import DataCleaningAnalysis

# --- Home Tab ---
def home_content():
    return """
# 📊 Google Analytics Data Cleaning & Visualization

Professional data wrangling & business intelligence project

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
"""

# --- Data Overview Tab ---
def data_overview(file):
    if file is None:
        return None, "", "", "", ""
    
    df = pd.read_csv(file.name)
    
    total_records = f"{len(df):,}"
    total_columns = str(len(df.columns))
    memory_usage = f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    missing_values = f"{df.isnull().sum().sum():,}"
    
    preview = df.head(10)
    
    return preview, total_records, total_columns, memory_usage, missing_values

def data_types_info(file):
    if file is None:
        return None
    
    df = pd.read_csv(file.name)
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values
    })
    return info_df

# --- Data Cleaning Tab ---
def run_cleaning_pipeline(file):
    if file is None:
        return None, None, "", ""
    
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_input:
        with open(file.name, 'rb') as f:
            tmp_input.write(f.read())
        tmp_input_path = tmp_input.name
    
    # Create temp output paths
    tmp_cleaned_path = tempfile.mktemp(suffix='_cleaned.csv')
    tmp_quality_path = tempfile.mktemp(suffix='_quality_report.csv')
    
    try:
        cleaner = DataCleaningAnalysis(data_path=tmp_input_path)
        cleaner.load_data()
        original_count = len(cleaner.df)
        
        cleaner.initial_assessment()
        cleaner.remove_duplicates()
        cleaner.handle_missing_values()
        cleaner.remove_bot_traffic()
        cleaner.normalize_timezones()
        cleaner.standardize_columns()
        cleaner.quality_report()
        
        cleaner.cleaned_df = cleaner.df
        cleaner.cleaned_df.to_csv(tmp_cleaned_path, index=False)
        cleaner.generate_quality_report_csv(output_path=tmp_quality_path)
        
        cleaned_count = len(cleaner.cleaned_df)
        
        return tmp_cleaned_path, tmp_quality_path, f"{original_count:,}", f"{cleaned_count:,}"
    except Exception as e:
        return None, None, f"Error: {str(e)}", ""
    finally:
        if os.path.exists(tmp_input_path):
            os.unlink(tmp_input_path)

# --- Visualizations Tab ---
def get_numeric_columns(file):
    if file is None:
        return gr.update(choices=[], value=None)
    
    df = pd.read_csv(file.name)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        return gr.update(choices=numeric_cols, value=numeric_cols[0])
    return gr.update(choices=[], value=None)

def visualize_column(file, column):
    if file is None or column is None:
        return None, "", "", "", ""
    
    df = pd.read_csv(file.name)
    
    if column not in df.columns:
        return None, "", "", "", ""
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    df[column].hist(bins=50, ax=ax, edgecolor='black', color='#1f77b4')
    ax.set_title(f'Distribution of {column}', fontsize=14)
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    
    # Statistics
    mean_val = f"{df[column].mean():.2f}"
    median_val = f"{df[column].median():.2f}"
    std_val = f"{df[column].std():.2f}"
    min_max_val = f"{df[column].min():.2f} / {df[column].max():.2f}"
    
    return fig, mean_val, median_val, std_val, min_max_val

# --- Quality Report Tab ---
def display_quality_report(file):
    if file is None:
        return None, "", "", ""
    
    df = pd.read_csv(file.name)
    
    passed = len(df[df['status'] == 'PASS'])
    warnings = len(df[df['status'] == 'WARNING'])
    failed = len(df[df['status'] == 'FAIL'])
    
    return df, str(passed), str(warnings), str(failed)

# --- Footer ---
footer_html = """
<div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #ddd;">
    <h4>💰 You can help me by Donating</h4>
    <div style="display: flex; justify-content: center; gap: 20px; margin: 15px 0;">
        <a href="https://www.buymeacoffee.com/nsrawat" target="_blank" style="background-color: #FFDD00; color: #000; padding: 12px 24px; border-radius: 5px; text-decoration: none; font-weight: bold;">☕ Buy Me a Coffee</a>
        <a href="https://paypal.me/NRawat710" target="_blank" style="background-color: #00457C; color: #fff; padding: 12px 24px; border-radius: 5px; text-decoration: none; font-weight: bold;">💳 PayPal</a>
    </div>
    <hr style="margin: 20px 0;">
    <p>Made with ❤️ by <a href="https://nsrawat.in" target="_blank" style="color: #1f77b4;">N S Rawat</a></p>
</div>
"""

# --- Build Gradio App ---
with gr.Blocks(title="Google Analytics Data Cleaning & Visualization", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Google Analytics Data Cleaning & Visualization")
    gr.Markdown("Professional data wrangling & business intelligence project")
    
    with gr.Tabs():
        # Home Tab
        with gr.Tab("🏠 Home"):
            gr.Markdown(home_content())
        
        # Data Overview Tab
        with gr.Tab("📈 Data Overview"):
            with gr.Row():
                overview_file = gr.File(label="Upload Google Analytics CSV file", file_types=[".csv"])
            
            with gr.Row():
                total_records = gr.Textbox(label="Total Records", interactive=False)
                total_columns = gr.Textbox(label="Total Columns", interactive=False)
                memory_usage = gr.Textbox(label="Memory Usage", interactive=False)
                missing_values = gr.Textbox(label="Missing Values", interactive=False)
            
            gr.Markdown("### Data Preview")
            overview_preview = gr.Dataframe(label="First 10 Rows")
            
            gr.Markdown("### Data Types")
            overview_types = gr.Dataframe(label="Column Information")
            
            overview_file.change(
                fn=data_overview,
                inputs=[overview_file],
                outputs=[overview_preview, total_records, total_columns, memory_usage, missing_values]
            )
            overview_file.change(
                fn=data_types_info,
                inputs=[overview_file],
                outputs=[overview_types]
            )
        
        # Data Cleaning Tab
        with gr.Tab("🧹 Data Cleaning"):
            with gr.Row():
                cleaning_file = gr.File(label="Upload CSV file for cleaning", file_types=[".csv"])
            
            run_btn = gr.Button("🚀 Run Cleaning Pipeline", variant="primary")
            
            with gr.Row():
                original_records = gr.Textbox(label="Original Records", interactive=False)
                cleaned_records = gr.Textbox(label="Cleaned Records", interactive=False)
            
            with gr.Row():
                cleaned_file_output = gr.File(label="Download Cleaned Data")
                quality_file_output = gr.File(label="Download Quality Report")
            
            run_btn.click(
                fn=run_cleaning_pipeline,
                inputs=[cleaning_file],
                outputs=[cleaned_file_output, quality_file_output, original_records, cleaned_records]
            )
        
        # Visualizations Tab
        with gr.Tab("📊 Visualizations"):
            with gr.Row():
                viz_file = gr.File(label="Upload cleaned CSV file", file_types=[".csv"])
            
            column_dropdown = gr.Dropdown(label="Select column to visualize", choices=[], interactive=True)
            
            viz_plot = gr.Plot(label="Distribution")
            
            with gr.Row():
                mean_box = gr.Textbox(label="Mean", interactive=False)
                median_box = gr.Textbox(label="Median", interactive=False)
                std_box = gr.Textbox(label="Std Dev", interactive=False)
                minmax_box = gr.Textbox(label="Min / Max", interactive=False)
            
            viz_file.change(
                fn=get_numeric_columns,
                inputs=[viz_file],
                outputs=[column_dropdown]
            )
            column_dropdown.change(
                fn=visualize_column,
                inputs=[viz_file, column_dropdown],
                outputs=[viz_plot, mean_box, median_box, std_box, minmax_box]
            )
        
        # Quality Report Tab
        with gr.Tab("📋 Quality Report"):
            with gr.Row():
                quality_file = gr.File(label="Upload data quality report CSV", file_types=[".csv"])
            
            quality_df = gr.Dataframe(label="Quality Report")
            
            with gr.Row():
                passed_box = gr.Textbox(label="✅ Passed", interactive=False)
                warnings_box = gr.Textbox(label="⚠️ Warnings", interactive=False)
                failed_box = gr.Textbox(label="❌ Failed", interactive=False)
            
            quality_file.change(
                fn=display_quality_report,
                inputs=[quality_file],
                outputs=[quality_df, passed_box, warnings_box, failed_box]
            )
    
    gr.HTML(footer_html)

if __name__ == "__main__":
    demo.launch()
