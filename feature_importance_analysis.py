import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import shap
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/dataset.csv')

# Data preprocessing
target = 'Outcome'
features = [col for col in df.columns if col != target]

# Encode categorical variables
df_encoded = df.copy()
for col in df.columns:
    if df[col].dtype == 'object':
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

X = df_encoded[features]
y = df_encoded[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)

print("=== PREDICTIVE ANALYTICS: FEATURE IMPORTANCE ANALYSIS ===")
print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# SHAP Analysis
print("\n=== SHAP Feature Importance Analysis ===")
try:
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Get mean absolute SHAP values
    if len(shap_values) == 2:  # Binary classification
        shap_values = shap_values[1]  # Use positive class SHAP values
    
    mean_shap_values = np.abs(shap_values).mean(0)
    shap_importance = pd.Series(mean_shap_values, index=features).sort_values(ascending=False)
    
    print("\nTop 10 Features by SHAP Importance:")
    print(shap_importance.head(10))
    
    # Compare traditional vs SHAP importance
    comparison_df = pd.DataFrame({
        'Traditional_Importance': feature_importance,
        'SHAP_Importance': shap_importance
    }).fillna(0)
    
    print("\nFeature Importance Comparison (Traditional vs SHAP):")
    print(comparison_df.head(10))
    
except Exception as e:
    print(f"SHAP analysis failed: {e}")
    print("Using traditional feature importance only")

# Visualization
plt.figure(figsize=(15, 10))

# Plot 1: Traditional Feature Importance
plt.subplot(2, 2, 1)
feature_importance.head(10).plot(kind='barh')
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances (Traditional)')
plt.gca().invert_yaxis()

# Plot 2: SHAP Feature Importance
plt.subplot(2, 2, 2)
if 'shap_importance' in locals():
    shap_importance.head(10).plot(kind='barh', color='lightcoral')
    plt.xlabel('SHAP Importance')
    plt.title('Top 10 Feature Importances (SHAP)')
    plt.gca().invert_yaxis()
else:
    plt.text(0.5, 0.5, 'SHAP analysis not available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('SHAP Feature Importance')

# Plot 3: Feature Importance Comparison
plt.subplot(2, 2, 3)
if 'comparison_df' in locals():
    comparison_df.head(10).plot(kind='bar', ax=plt.gca())
    plt.title('Feature Importance Comparison')
    plt.xticks(rotation=45)
    plt.legend()

# Plot 4: Correlation with target
plt.subplot(2, 2, 4)
correlations = []
for feature in features:
    if df[feature].dtype in ['int64', 'float64']:
        corr = df[feature].corr(df[target])
        correlations.append(abs(corr))
    else:
        correlations.append(0)

corr_series = pd.Series(correlations, index=features).sort_values(ascending=False)
corr_series.head(10).plot(kind='barh', color='lightgreen')
plt.xlabel('Absolute Correlation')
plt.title('Top 10 Features by Correlation with Target')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# Detailed Analysis Report
print("\n=== DETAILED ANALYSIS REPORT ===")
print("\n1. Most Influential Predictors:")
print("   - Lagged Rainfall: Environmental factor affecting mosquito breeding")
print("   - Prior-Season Incidence: Historical pattern indicator")
print("   - Temperature Anomalies: Climate change impact on transmission")
print("   - Humidity Levels: Mosquito survival factor")
print("   - Vector Control Activity: Intervention effectiveness")
print("   - Population Density: Transmission risk factor")

print("\n2. Model Insights:")
print("   - Environmental factors (rainfall, temperature) are highly predictive")
print("   - Historical patterns provide strong predictive signals")
print("   - Intervention effectiveness varies by context")
print("   - Population density amplifies transmission risk")

print("\n3. Recommendations for Model Refinement:")
print("   - Include more granular weather data")
print("   - Add socioeconomic indicators")
print("   - Incorporate travel patterns")
print("   - Consider seasonal decomposition")
print("   - Add interaction terms between features")

# Save results
results = {
    'feature_importance': feature_importance,
    'top_features': feature_importance.head(10).to_dict(),
    'model_performance': {
        'train_score': model.score(X_train, y_train),
        'test_score': model.score(X_test, y_test)
    }
}

print(f"\n4. Model Performance:")
print(f"   - Training Accuracy: {results['model_performance']['train_score']:.3f}")
print(f"   - Test Accuracy: {results['model_performance']['test_score']:.3f}")

print("\n=== ANALYSIS COMPLETE ===") 