-- Google Analytics Data Analysis Queries
-- This file contains SQL queries for analyzing cleaned Google Analytics data

-- ============================================
-- 1. Session Overview Analysis
-- ============================================

-- Total sessions by date
SELECT 
    DATE(timestamp) as session_date,
    COUNT(*) as total_sessions,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(session_duration) as avg_session_duration,
    SUM(pageviews) as total_pageviews
FROM cleaned_data
GROUP BY DATE(timestamp)
ORDER BY session_date DESC;

-- ============================================
-- 2. Traffic Source Analysis
-- ============================================

-- Sessions by traffic source and medium
SELECT 
    traffic_source,
    medium,
    COUNT(*) as sessions,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(bounce_rate) as avg_bounce_rate,
    AVG(session_duration) as avg_duration,
    SUM(revenue) as total_revenue
FROM cleaned_data
GROUP BY traffic_source, medium
ORDER BY sessions DESC;

-- ============================================
-- 3. Geographic Analysis
-- ============================================

-- Top countries by sessions and revenue
SELECT 
    country,
    COUNT(*) as sessions,
    COUNT(DISTINCT user_id) as unique_users,
    SUM(revenue) as total_revenue,
    AVG(session_duration) as avg_duration,
    AVG(bounce_rate) as avg_bounce_rate
FROM cleaned_data
GROUP BY country
ORDER BY total_revenue DESC
LIMIT 20;

-- ============================================
-- 4. Device & Browser Analysis
-- ============================================

-- Sessions by device type
SELECT 
    device_type,
    browser,
    operating_system,
    COUNT(*) as sessions,
    AVG(session_duration) as avg_duration,
    AVG(bounce_rate) as avg_bounce_rate,
    SUM(revenue) as total_revenue
FROM cleaned_data
GROUP BY device_type, browser, operating_system
ORDER BY sessions DESC;

-- ============================================
-- 5. User Behavior Analysis
-- ============================================

-- User engagement metrics by session duration buckets
SELECT 
    CASE 
        WHEN session_duration < 30 THEN '0-30s'
        WHEN session_duration < 60 THEN '30-60s'
        WHEN session_duration < 180 THEN '1-3min'
        WHEN session_duration < 600 THEN '3-10min'
        ELSE '10min+'
    END as duration_bucket,
    COUNT(*) as sessions,
    AVG(pageviews) as avg_pageviews,
    AVG(bounce_rate) as avg_bounce_rate,
    SUM(revenue) as total_revenue
FROM cleaned_data
GROUP BY duration_bucket
ORDER BY MIN(session_duration);

-- ============================================
-- 6. E-commerce Analysis
-- ============================================

-- Revenue analysis by product category
SELECT 
    product_category,
    COUNT(*) as transactions,
    SUM(revenue) as total_revenue,
    AVG(revenue) as avg_transaction_value,
    SUM(items) as total_items_sold
FROM cleaned_data
WHERE revenue > 0
GROUP BY product_category
ORDER BY total_revenue DESC;

-- ============================================
-- 7. Time-based Analysis
-- ============================================

-- Sessions by hour of day
SELECT 
    EXTRACT(HOUR FROM timestamp) as hour_of_day,
    COUNT(*) as sessions,
    AVG(session_duration) as avg_duration,
    SUM(revenue) as total_revenue
FROM cleaned_data
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- Sessions by day of week
SELECT 
    EXTRACT(DAYOFWEEK FROM timestamp) as day_of_week,
    CASE EXTRACT(DAYOFWEEK FROM timestamp)
        WHEN 1 THEN 'Sunday'
        WHEN 2 THEN 'Monday'
        WHEN 3 THEN 'Tuesday'
        WHEN 4 THEN 'Wednesday'
        WHEN 5 THEN 'Thursday'
        WHEN 6 THEN 'Friday'
        WHEN 7 THEN 'Saturday'
    END as day_name,
    COUNT(*) as sessions,
    AVG(session_duration) as avg_duration,
    SUM(revenue) as total_revenue
FROM cleaned_data
GROUP BY day_of_week, day_name
ORDER BY day_of_week;

-- ============================================
-- 8. User Segmentation
-- ============================================

-- New vs Returning users
SELECT 
    user_type,
    COUNT(*) as sessions,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(session_duration) as avg_duration,
    AVG(pageviews) as avg_pageviews,
    SUM(revenue) as total_revenue
FROM cleaned_data
GROUP BY user_type;

-- ============================================
-- 9. Conversion Funnel Analysis
-- ============================================

-- Conversion funnel by traffic source
SELECT 
    traffic_source,
    COUNT(*) as total_sessions,
    SUM(CASE WHEN pageviews > 0 THEN 1 ELSE 0 END) as sessions_with_pageviews,
    SUM(CASE WHEN session_duration > 60 THEN 1 ELSE 0 END) as engaged_sessions,
    SUM(CASE WHEN revenue > 0 THEN 1 ELSE 0 END) as converting_sessions,
    SUM(revenue) as total_revenue,
    ROUND(100.0 * SUM(CASE WHEN revenue > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as conversion_rate
FROM cleaned_data
GROUP BY traffic_source
ORDER BY total_revenue DESC;

-- ============================================
-- 10. Data Quality Validation Queries
-- ============================================

-- Check for any remaining duplicates
SELECT 
    session_id,
    COUNT(*) as duplicate_count
FROM cleaned_data
GROUP BY session_id
HAVING COUNT(*) > 1;

-- Check for missing critical values
SELECT 
    'Missing user_id' as issue,
    COUNT(*) as count
FROM cleaned_data
WHERE user_id IS NULL
UNION ALL
SELECT 
    'Missing timestamp' as issue,
    COUNT(*) as count
FROM cleaned_data
WHERE timestamp IS NULL
UNION ALL
SELECT 
    'Missing session_id' as issue,
    COUNT(*) as count
FROM cleaned_data
WHERE session_id IS NULL;

-- Data completeness check
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT session_id) as unique_sessions,
    MIN(timestamp) as earliest_date,
    MAX(timestamp) as latest_date,
    COUNT(CASE WHEN revenue > 0 THEN 1 END) as transactions,
    SUM(revenue) as total_revenue
FROM cleaned_data;

