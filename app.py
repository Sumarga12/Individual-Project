import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA  # <-- Add this import

st.set_page_config(
    page_title="Dengue Outbreak Analytics Dashboard",
    page_icon="🦟",
    layout="wide"
)

# --- STYLE ---
st.markdown("""
    <style>
    body, .stApp, .block-container, .main, [data-testid="stSidebar"], [data-testid="stHeader"] {
        background-color: #fff !important;
        color: #111 !important;
    }
    label, div[data-testid="stSelectboxLabel"], div[data-testid="stFormLabel"],
    div[data-testid="stSliderLabel"], div[data-testid="stTextInputLabel"],
    div[data-testid="stNumberInputLabel"], div[data-testid="stRadioLabel"],
    div[data-testid="stDateInputLabel"], div[data-testid="stTimeInputLabel"] {
        color: #111 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    .stDataFrame, .stTable {
        background-color: #fff !important;
        color: #111 !important;
    }
    .stDownloadButton button {
        background-color: #fff !important;
        color: #111 !important;
        border: 1.5px solid #0d253f !important;
        font-weight: 600 !important;
        font-size: 1.02rem !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
        transition: background 0.25s, color 0.25s;
    }
    .stDownloadButton button:hover {
        background-color: #e0e7ef !important;
        color: #0d253f !important;
        border: 1.5px solid #0d253f !important;
    }
    /* Make KPI metric values black for better visibility */
    div[data-testid="stMetricValue"] {
        color: #111 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/dataset.csv')
    return df

df = load_data()

# --- DATA PREP ---
bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
if 'Date' not in df.columns:
    np.random.seed(42)
    df['Date'] = pd.to_datetime('2024-06-01') + pd.to_timedelta(np.random.randint(0, 60, size=len(df)), unit='D')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M').astype(str)
outcome_map = {1: 'Positive', 0: 'Negative'}
df['OutcomeLabel'] = df['Outcome'].map(outcome_map)

# --- FILTERS ---
gender_options = ['All'] + df['Gender'].dropna().unique().tolist()
outcome_options = ['All', 'Positive', 'Negative']
district_options = ['All'] + df['District'].dropna().unique().tolist()

filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    gender_filter = st.selectbox('Select Gender', options=gender_options)
with filter_col2:
    outcome_filter = st.selectbox('Select Outcome', options=outcome_options)
with filter_col3:
    district_filter = st.selectbox('Select District', options=district_options)

filtered_df = df.copy()
if gender_filter != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]
if outcome_filter != 'All':
    filtered_df = filtered_df[filtered_df['OutcomeLabel'] == outcome_filter]
if district_filter != 'All':
    filtered_df = filtered_df[filtered_df['District'] == district_filter]

# --- EXPORT BUTTON ---
st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_dengue_data.csv",
    mime="text/csv"
)

# --- KPIs ---
num_total = filtered_df.shape[0]
num_positive = filtered_df[filtered_df['OutcomeLabel'] == 'Positive'].shape[0]
num_negative = filtered_df[filtered_df['OutcomeLabel'] == 'Negative'].shape[0]
positive_rate = 100 * num_positive / num_total if num_total > 0 else 0
avg_age = filtered_df['Age'].mean() if num_total > 0 else 0

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
kpi_col1.metric("Total Patients", num_total)
kpi_col2.metric("Average Age", f"{avg_age:.1f}")
kpi_col3.metric("Positive Cases", num_positive)
kpi_col4.metric("Negative Cases", num_negative)
kpi_col5.metric("Positive Rate (%)", f"{positive_rate:.1f}")

# --- ALERT BANNER ---
if positive_rate > 20:
    st.markdown(
        f"""
        <div style="border-left: 8px solid #d90429; background-color: #ffe6e6; padding: 1.2em 1em; margin: 1.2em 0; border-radius: 0.6em;">
        <b style="color:#d90429; font-size:1.1em;">🚨 WARNING:</b> 
        <span style="color:#111;">
        Out of every 100 people checked, <b>{positive_rate:.1f}</b> have dengue. This is higher than the safe level (20 out of 100).<br><br>
        <b>Please act quickly to control the outbreak!</b>
        </span>
        </div>
        """, unsafe_allow_html=True)

# --- DASHBOARD PANELS ---
st.title("Dengue Outbreak Analytics Dashboard")
st.markdown("*Predictive and Prescriptive Analytics for Early Detection and Control of Dengue Outbreaks: A strategic guide for government officials.*")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# 1. Feature Importance (Predictive)
target = 'Outcome'
features = [col for col in filtered_df.columns if col not in [target, 'OutcomeLabel', 'Date', 'Month', 'AgeGroup']]
df_encoded = filtered_df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
X = df_encoded[features]
y = df_encoded[target]
if len(X) > 0 and len(y.unique()) > 1:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance = pd.Series(importances, index=features).sort_values(ascending=False).head(8)
    fig1 = go.Figure(go.Bar(
        x=feature_importance.values[::-1],
        y=feature_importance.index[::-1],
        orientation='h',
        marker_color='royalblue'
    ))
    fig1.update_layout(title="Top Predictive Features for Severe Dengue", xaxis_title="Importance", yaxis_title="Feature")
    col1.plotly_chart(fig1, use_container_width=True)
else:
    col1.info("Not enough data for feature importance.")

# 2. Cases Over Time (Early Detection)
cases_per_month = filtered_df.groupby(filtered_df['Date'].dt.to_period('M').astype(str)).size()
severe_per_month = filtered_df[filtered_df['Outcome'] == 1].groupby(filtered_df['Date'].dt.to_period('M').astype(str)).size()
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=cases_per_month.index, y=cases_per_month.values, mode='lines+markers', name='All Cases', line=dict(color='darkorange')))
fig2.add_trace(go.Scatter(x=severe_per_month.index, y=severe_per_month.values, mode='lines+markers', name='Severe Cases', line=dict(color='red')))

# --- Forecasting (ARIMA) ---
try:
    # Prepare the time series (monthly, sorted)
    ts = cases_per_month.sort_index()
    ts.index = pd.to_datetime(ts.index)
    # Fit ARIMA (auto order for simplicity, or (1,1,1) as a default)
    model = ARIMA(ts, order=(1,1,1))
    model_fit = model.fit()
    # Forecast next 3 months
    forecast_steps = 3
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(), periods=forecast_steps, freq='MS')
    # Add forecast to chart
    fig2.add_trace(go.Scatter(
        x=forecast_index.strftime('%Y-%m'),
        y=forecast.values,
        mode='lines+markers',
        name='Forecast (next 3 months)',
        line=dict(color='royalblue', dash='dash')
    ))
except Exception as e:
    fig2.add_annotation(text=f"Forecasting error: {e}", xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False, font=dict(color="red"))

fig2.update_layout(title="Dengue Cases Over Time (with Forecast)", xaxis_title="Month", yaxis_title="Number of Cases")
col2.plotly_chart(fig2, use_container_width=True)

# 3. Cases by District (Targeted Control)
cases_per_district = filtered_df['District'].value_counts().head(10)
fig3 = go.Figure(go.Bar(
    x=cases_per_district.index,
    y=cases_per_district.values,
    marker_color='seagreen'
))
fig3.update_layout(title="Top 10 Districts by Dengue Cases", xaxis_title="District", yaxis_title="Number of Cases")
col3.plotly_chart(fig3, use_container_width=True)

# 4. Intervention Effectiveness (Prescriptive)
vc_outcome = filtered_df.groupby('Vector_Control_Activity')['Outcome'].mean()
ac_outcome = filtered_df.groupby('Awareness_Campaign')['Outcome'].mean()
fig4 = go.Figure()
fig4.add_trace(go.Bar(x=vc_outcome.index, y=vc_outcome.values, name='Vector Control', marker_color='skyblue'))
fig4.add_trace(go.Bar(x=ac_outcome.index, y=ac_outcome.values, name='Awareness Campaign', marker_color='orange'))
fig4.update_layout(
    title="Intervention Effectiveness (Severe Outcome Rate)",
    xaxis_title="Intervention",
    yaxis_title="Severe Outcome Rate",
    barmode='group'
)
col4.plotly_chart(fig4, use_container_width=True)

# --- NEW: DISTRICT COMPARISON CHARTS ---
st.markdown("---")
st.subheader("📊 District Comparison Analysis")

# Create district comparison metrics
district_comparison = filtered_df.groupby('District').agg({
    'Outcome': ['count', 'mean', 'sum'],
    'Age': 'mean',
    'Rainfall_mm': 'mean',
    'Temperature_C': 'mean',
    'Humidity_%': 'mean',
    'Vector_Control_Activity': lambda x: (x == 'Yes').mean(),
    'Awareness_Campaign': lambda x: (x == 'Yes').mean()
}).round(3)

district_comparison.columns = ['Total_Cases', 'Severe_Rate', 'Severe_Cases', 'Avg_Age', 'Avg_Rainfall', 'Avg_Temperature', 'Avg_Humidity', 'VC_Coverage', 'AC_Coverage']
district_comparison = district_comparison.sort_values('Total_Cases', ascending=False)

# District comparison charts
comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    # Cases vs Severe Rate Scatter Plot
    fig_comp1 = go.Figure()
    fig_comp1.add_trace(go.Scatter(
        x=district_comparison['Total_Cases'],
        y=district_comparison['Severe_Rate'],
        mode='markers+text',
        text=district_comparison.index,
        textposition="top center",
        marker=dict(
            size=district_comparison['Total_Cases']/10,
            color=district_comparison['Severe_Rate'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Severe Rate")
        ),
        hovertemplate='<b>%{text}</b><br>' +
                     'Total Cases: %{x}<br>' +
                     'Severe Rate: %{y:.3f}<br>' +
                     '<extra></extra>'
    ))
    fig_comp1.update_layout(
        title="District Risk Analysis: Cases vs Severe Rate",
        xaxis_title="Total Cases",
        yaxis_title="Severe Outcome Rate",
        height=500
    )
    st.plotly_chart(fig_comp1, use_container_width=True)

with comp_col2:
    # Environmental Factors Comparison
    fig_comp2 = go.Figure()
    
    # Normalize values for better comparison
    normalized_data = district_comparison[['Avg_Rainfall', 'Avg_Temperature', 'Avg_Humidity']].copy()
    for col in normalized_data.columns:
        normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / (normalized_data[col].max() - normalized_data[col].min())
    
    fig_comp2.add_trace(go.Scatter(
        x=normalized_data.index,
        y=normalized_data['Avg_Rainfall'],
        mode='lines+markers',
        name='Rainfall',
        line=dict(color='blue', width=3)
    ))
    fig_comp2.add_trace(go.Scatter(
        x=normalized_data.index,
        y=normalized_data['Avg_Temperature'],
        mode='lines+markers',
        name='Temperature',
        line=dict(color='red', width=3)
    ))
    fig_comp2.add_trace(go.Scatter(
        x=normalized_data.index,
        y=normalized_data['Avg_Humidity'],
        mode='lines+markers',
        name='Humidity',
        line=dict(color='green', width=3)
    ))
    
    fig_comp2.update_layout(
        title="Environmental Factors by District (Normalized)",
        xaxis_title="District",
        yaxis_title="Normalized Values",
        height=500,
        xaxis=dict(tickangle=45)
    )
    st.plotly_chart(fig_comp2, use_container_width=True)





# --- Enhanced Visualizations: Histograms and Boxplots ---
st.markdown("---")
st.subheader("Distribution of Numeric Features")

num_features = ['Age', 'Rainfall_mm', 'Temperature_C', 'Humidity_%']
num_col1, num_col2 = st.columns(2)

with num_col1:
    feature = st.selectbox('Select feature for Histogram', num_features, key='hist_feature')
    
    # Define distinct colors for each feature
    color_map = {
        'Age': '#FF6B6B',        # Red
        'Rainfall_mm': '#4ECDC4', # Teal
        'Temperature_C': '#45B7D1', # Blue
        'Humidity_%': '#87CEEB'   # Light blue (keeping current color for humidity)
    }
    
    # Get the color for the selected feature
    selected_color = color_map.get(feature, '#96CEB4')  # Default to green if feature not found
    
    fig_hist = px.histogram(
        filtered_df, 
        x=feature, 
        nbins=30, 
        title=f'Histogram of {feature}',
        color_discrete_sequence=[selected_color]
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with num_col2:
    feature_box = st.selectbox('Select feature for Boxplot', num_features, key='box_feature')
    fig_box = px.box(filtered_df, y=feature_box, title=f'Boxplot of {feature_box}')
    st.plotly_chart(fig_box, use_container_width=True)

# --- Distribution of Cases by Month and Age Group ---
st.markdown("---")
st.subheader("Distribution of Cases by Month and Age Group")

# Distribution by Month
cases_by_month = df.groupby(df['Date'].dt.to_period('M').astype(str)).size()
fig_month = px.bar(x=cases_by_month.index, y=cases_by_month.values, labels={'x': 'Month', 'y': 'Number of Cases'}, title='Cases by Month', color_discrete_sequence=['darkorange'])
st.plotly_chart(fig_month, use_container_width=True)

# Distribution by Age Group
cases_by_agegroup = df['AgeGroup'].value_counts().sort_index()
fig_agegroup = px.bar(x=cases_by_agegroup.index, y=cases_by_agegroup.values, labels={'x': 'Age Group', 'y': 'Number of Cases'}, title='Cases by Age Group', color_discrete_sequence=['darkorange'])
st.plotly_chart(fig_agegroup, use_container_width=True)

# --- Geographic Distribution: Cases by District ---
st.markdown("---")
st.subheader("Geographic Distribution: Cases by District")

# Map-like bar chart for districts
cases_by_district = df['District'].value_counts().sort_values(ascending=False)
fig_district = px.bar(x=cases_by_district.index, y=cases_by_district.values, labels={'x': 'District', 'y': 'Number of Cases'}, title='Cases by District', color_discrete_sequence=['mediumseagreen'])
st.plotly_chart(fig_district, use_container_width=True)

# --- Calculate Hotspot Table Data (district_stats) ---
district_stats = (
    filtered_df.groupby('District')
    .agg(total=('Outcome','count'), positives=('Outcome','sum'))
    .assign(positive_rate=lambda x: 100 * x['positives'] / x['total'])
    .sort_values('positive_rate', ascending=False)
    .head(5)
    .reset_index()
)

# --- Policy Recommendations Panel ---
st.markdown("---")
st.subheader("Policy Recommendations for Government Officials")

recommendations_html = []
if positive_rate > 20:
    recommendations_html.append(f"""
    <div style='background:#ffe6e6; border-left:6px solid #d90429; padding:1em; margin-bottom:1em; border-radius:0.5em;'>
        <span style='font-size:1.3em;'>[ALERT] <b>High positive rate detected ({positive_rate:.1f}%).</b></span><br>
        <span style='font-size:1.1em;'><b>Initiate outbreak response protocols immediately.</b></span>
    </div>""")
if not district_stats.empty:
    top_district = district_stats.iloc[0]['District']
    recommendations_html.append(f"""
    <div style='background:#fff4e6; border-left:6px solid #ff8800; padding:1em; margin-bottom:1em; border-radius:0.5em;'>
        <span style='font-size:1.2em;'>[HOTSPOT] <b>Hotspot Alert:</b></span> Focus vector control and awareness campaigns in <b>{top_district}</b>.
    </div>""")
# Find most effective intervention
try:
    best_vc = vc_outcome.idxmin() if not vc_outcome.empty else None
    best_ac = ac_outcome.idxmin() if not ac_outcome.empty else None
    if best_vc and best_ac:
        if vc_outcome[best_vc] < ac_outcome[best_ac]:
            recommendations_html.append(f"""
            <div style='background:#e6f7ff; border-left:6px solid #0077b6; padding:1em; margin-bottom:1em; border-radius:0.5em;'>
                <span style='font-size:1.2em;'>[VECTOR CONTROL]</span> Vector control activity is effective. Continue and prioritize this intervention.
            </div>""")
        else:
            recommendations_html.append(f"""
            <div style='background:#f3e6ff; border-left:6px solid #a259d9; padding:1em; margin-bottom:1em; border-radius:0.5em;'>
                <span style='font-size:1.2em;'>[AWARENESS] <b>Awareness Campaign '{best_ac}'</b></span> is associated with the lowest severe outcome rate. <b>Prioritize this intervention.</b>
            </div>""")
except Exception:
    pass
if not recommendations_html:
    recommendations_html.append("""
    <div style='background:#e6ffe6; border-left:6px solid #2ecc40; padding:1em; margin-bottom:1em; border-radius:0.5em;'>
        <span style='font-size:1.2em;'>[OK] <b>No urgent actions required.</b></span> Continue monitoring and routine interventions.
    </div>""")
st.markdown(''.join(recommendations_html), unsafe_allow_html=True)

# --- Hotspot Table ---
st.subheader("Top 5 Districts by Positive Rate (Hotspot Alert)")
district_stats.index = district_stats.index + 1  # Start index from 1 instead of 0
st.dataframe(district_stats)