"""
Gradio App for Google Analytics Data Cleaning & Visualization
Mirrors the functionality of streamlit_app.py
Deploy on Hugging Face Spaces
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
from datetime import datetime

# --- DataCleaningAnalysis class (embedded for HF Spaces) ---
class DataCleaningAnalysis:
    def __init__(self, data_path='data/raw/google_analytics_export.csv'):
        self.data_path = data_path
        self.df = None
        self.cleaned_df = None
        
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def initial_assessment(self):
        pass
    
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self.df
    
    def handle_missing_values(self):
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['float64', 'int64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna('Unknown', inplace=True)
        return self.df
    
    def remove_bot_traffic(self):
        user_agent_col = None
        for col in self.df.columns:
            if 'user_agent' in col.lower() or 'useragent' in col.lower():
                user_agent_col = col
                break
        
        if user_agent_col:
            bot_keywords = ['bot', 'crawler', 'spider', 'googlebot']
            mask = ~self.df[user_agent_col].astype(str).str.lower().str.contains('|'.join(bot_keywords), na=False)
            self.df = self.df[mask]
        return self.df
    
    def normalize_timezones(self):
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], utc=True)
        return self.df
    
    def standardize_columns(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].str.strip().str.title()
        return self.df
    
    def quality_report(self):
        return self.df
    
    def generate_quality_report_csv(self, output_path='quality_report.csv'):
        quality_metrics = []
        total_rows = len(self.df)
        total_cols = len(self.df.columns)
        missing_total = self.df.isnull().sum().sum()
        duplicates = self.df.duplicated().sum()
        
        quality_metrics.append({'metric': 'Total Records', 'value': total_rows, 'percentage': 100.0, 'status': 'PASS'})
        quality_metrics.append({'metric': 'Total Columns', 'value': total_cols, 'percentage': 100.0, 'status': 'PASS'})
        quality_metrics.append({'metric': 'Missing Values', 'value': missing_total, 'percentage': (missing_total / (total_rows * total_cols)) * 100 if total_rows > 0 else 0, 'status': 'PASS' if (missing_total / (total_rows * total_cols)) * 100 < 1 else 'WARNING'})
        quality_metrics.append({'metric': 'Duplicate Rows', 'value': duplicates, 'percentage': (duplicates / total_rows) * 100 if total_rows > 0 else 0, 'status': 'PASS' if duplicates == 0 else 'FAIL'})
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0
            quality_metrics.append({'metric': f'Column: {col}', 'value': f'Missing: {missing_count}', 'percentage': missing_pct, 'status': 'PASS' if missing_pct < 5 else 'WARNING' if missing_pct < 20 else 'FAIL'})
        
        passed = sum(1 for m in quality_metrics if m['status'] == 'PASS')
        total_checks = len(quality_metrics)
        quality_score = (passed / total_checks) * 100 if total_checks > 0 else 0
        quality_metrics.append({'metric': 'Overall Quality Score', 'value': f'{quality_score:.2f}%', 'percentage': quality_score, 'status': 'PASS' if quality_score >= 95 else 'WARNING' if quality_score >= 80 else 'FAIL'})
        
        report_df = pd.DataFrame(quality_metrics)
        report_df.to_csv(output_path, index=False)
        return report_df

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
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_input:
        with open(file.name, 'rb') as f:
            tmp_input.write(f.read())
        tmp_input_path = tmp_input.name
    
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df[column].hist(bins=50, ax=ax, edgecolor='black', color='#1f77b4')
    ax.set_title(f'Distribution of {column}', fontsize=14)
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    
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
        <a href="https://www.buymeacoffee.com/nsrawat?ref=HuggingFace" target="_blank" style="background-color: #FFDD00; color: #000; padding: 12px 24px; border-radius: 5px; text-decoration: none; font-weight: bold;">☕ Buy Me a Coffee</a>
        <a href="https://paypal.me/NRawat710?ref=HuggingFace" target="_blank" style="background-color: #00457C; color: #fff; padding: 12px 24px; border-radius: 5px; text-decoration: none; font-weight: bold;">💳 PayPal</a>
        <a href="https://withupi.com/@nsrawat?ref=HuggingFace" target="_blank" style="background-color: #4CAF50; color: #fff; padding: 12px 24px; border-radius: 5px; text-decoration: none; font-weight: bold;">₹ UPI</a>
    </div>
    <hr style="margin: 20px 0;">
    <p>Made with ❤️ by <a href="https://nsrawat.in" target="_blank" style="color: #1f77b4;">N S Rawat</a> | <a href="https://github.com/iNSRawat" target="_blank" style="color: #1f77b4;">GitHub</a> | <a href="https://github.com/iNSRawat/data-cleaning-visualization" target="_blank" style="color: #1f77b4;">Project Repo</a></p>
</div>
"""



# --- Build Gradio App ---
with gr.Blocks(title="Google Analytics Data Cleaning & Visualization", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Google Analytics Data Cleaning & Visualization")
    gr.Markdown("Professional data wrangling & business intelligence project")
    
    with gr.Tabs():
        with gr.Tab("🏠 Home"):
            gr.Markdown(home_content())
        
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
            
            overview_file.change(fn=data_overview, inputs=[overview_file], outputs=[overview_preview, total_records, total_columns, memory_usage, missing_values], api_name="data_overview")
            overview_file.change(fn=data_types_info, inputs=[overview_file], outputs=[overview_types], api_name="data_types")
        
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
            
            run_btn.click(fn=run_cleaning_pipeline, inputs=[cleaning_file], outputs=[cleaned_file_output, quality_file_output, original_records, cleaned_records], api_name="run_cleaning")
        
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
            
            viz_file.change(fn=get_numeric_columns, inputs=[viz_file], outputs=[column_dropdown], api_name="get_columns")
            column_dropdown.change(fn=visualize_column, inputs=[viz_file, column_dropdown], outputs=[viz_plot, mean_box, median_box, std_box, minmax_box], api_name="visualize")
        
        with gr.Tab("📋 Quality Report"):
            with gr.Row():
                quality_file = gr.File(label="Upload data quality report CSV", file_types=[".csv"])
            
            quality_df = gr.Dataframe(label="Quality Report")
            
            with gr.Row():
                passed_box = gr.Textbox(label="✅ Passed", interactive=False)
                warnings_box = gr.Textbox(label="⚠️ Warnings", interactive=False)
                failed_box = gr.Textbox(label="❌ Failed", interactive=False)
            
            quality_file.change(fn=display_quality_report, inputs=[quality_file], outputs=[quality_df, passed_box, warnings_box, failed_box], api_name="quality_report")
    
    gr.HTML(footer_html)

if __name__ == "__main__":
    demo.queue().launch()
