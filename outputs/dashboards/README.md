# Tableau Dashboards

This directory contains Tableau dashboard files for visualizing Google Analytics data.

## Expected Files

- `analytics_dashboard.twbx` - Main interactive dashboard for Google Analytics insights

## Dashboard Features

The Tableau dashboard includes:

1. **Session Overview**
   - Total sessions by date
   - Unique users over time
   - Average session duration trends

2. **Traffic Source Analysis**
   - Sessions by source and medium
   - Conversion rates by traffic source
   - Revenue attribution

3. **Geographic Analysis**
   - Top countries by sessions and revenue
   - Regional performance metrics
   - Geographic heat maps

4. **Device & Browser Analysis**
   - Device type breakdown
   - Browser and OS performance
   - Mobile vs Desktop comparison

5. **User Behavior**
   - Session duration distributions
   - Pageview patterns
   - Bounce rate analysis

6. **E-commerce Metrics**
   - Revenue trends
   - Product category performance
   - Transaction analysis

## Data Source

The dashboard connects to the cleaned data file:
- `data/processed/cleaned_data.csv`

## Usage

1. Open Tableau Desktop
2. Connect to the cleaned data CSV file
3. Build visualizations using the SQL queries in `outputs/sql_queries/analysis_queries.sql` as reference
4. Save the workbook as `analytics_dashboard.twbx` in this directory

## Notes

- The dashboard file (`.twbx`) is typically large and may not be included in version control
- Use Tableau Public or Tableau Server for sharing dashboards
- Ensure data privacy compliance when working with user session data

