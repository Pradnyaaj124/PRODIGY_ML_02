import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Mall_Customers.csv')

print("="*60)
print("CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING")
print("="*60)

print("\n📊 Dataset Info:")
print(df.info())

print("\n👥 First few rows:")
print(df.head())

print("\n📈 Statistical Summary:")
print(df.describe())

print("\n🔍 Missing Values:")
print(df.isnull().sum())

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

print("\n📋 Columns after cleaning:")
print(df.columns.tolist())

print("\n" + "="*60)
print("PART 1: CLUSTERING BASED ON ANNUAL INCOME & SPENDING SCORE")
print("="*60)

X1 = df[['annual_income_(k$)', 'spending_score_(1-100)']].values

scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)

inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X1_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X1_scaled, kmeans.labels_))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k')
plt.grid(True)

plt.tight_layout()
plt.savefig('elbow_method.png')
plt.show()

optimal_k = 5
print(f"\n✅ Using k = {optimal_k} clusters")

kmeans1 = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster_Income_Spending'] = kmeans1.fit_predict(X1_scaled)

centers_scaled = kmeans1.cluster_centers_
centers_original = scaler1.inverse_transform(centers_scaled)

print("\n📊 Cluster Centers (Annual Income, Spending Score):")
for i, center in enumerate(centers_original):
    print(f"   Cluster {i}: Income = ${center[0]:.1f}K, Spending Score = {center[1]:.1f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X1[:, 0], X1[:, 1], c=df['Cluster_Income_Spending'], 
                      cmap='viridis', alpha=0.6)
plt.scatter(centers_original[:, 0], centers_original[:, 1], 
            c='red', marker='X', s=200, edgecolors='black', linewidth=2, label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments (Income vs Spending)')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cluster_counts = df['Cluster_Income_Spending'].value_counts().sort_index()
colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
plt.bar(cluster_counts.index, cluster_counts.values, color=colors)
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Customer Distribution Across Clusters')
for i, v in enumerate(cluster_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('income_spending_clusters.png')
plt.show()

print("\n" + "="*60)
print("PART 2: CLUSTERING BASED ON AGE, INCOME & SPENDING SCORE")
print("="*60)

X2 = df[['age', 'annual_income_(k$)', 'spending_score_(1-100)']].values

scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

kmeans2 = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster_Age_Income_Spending'] = kmeans2.fit_predict(X2_scaled)

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['age'], df['annual_income_(k$)'], df['spending_score_(1-100)'],
                     c=df['Cluster_Age_Income_Spending'], cmap='viridis', s=50, alpha=0.6)

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score')
ax.set_title('Customer Segments in 3D (Age, Income, Spending)')

plt.colorbar(scatter, label='Cluster', pad=0.1)
plt.savefig('3d_clusters.png')
plt.show()

print("\n📊 Cluster Statistics:")
print("="*40)

for cluster in range(optimal_k):
    cluster_data = df[df['Cluster_Age_Income_Spending'] == cluster]
    print(f"\n🎯 Cluster {cluster}:")
    print(f"   Size: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"   Avg Age: {cluster_data['age'].mean():.1f} years")
    print(f"   Avg Income: ${cluster_data['annual_income_(k$)'].mean():.1f}K")
    print(f"   Avg Spending Score: {cluster_data['spending_score_(1-100)'].mean():.1f}")

print("\n" + "="*60)
print("🎯 CUSTOMER SEGMENT PROFILES")
print("="*60)

profiles = []
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster_Age_Income_Spending'] == cluster]
    
    age = cluster_data['age'].mean()
    income = cluster_data['annual_income_(k$)'].mean()
    spending = cluster_data['spending_score_(1-100)'].mean()
    
    if age < 30:
        age_cat = "Young"
    elif age < 45:
        age_cat = "Middle-aged"
    else:
        age_cat = "Senior"
    
    if income < 40:
        income_cat = "Low Income"
    elif income < 70:
        income_cat = "Middle Income"
    else:
        income_cat = "High Income"
    
    if spending < 40:
        spending_cat = "Low Spender"
    elif spending < 60:
        spending_cat = "Average Spender"
    else:
        spending_cat = "High Spender"
    
    profile = f"{age_cat} {income_cat} {spending_cat}"
    profiles.append(profile)
    
    print(f"\nCluster {cluster}: {profile}")
    print(f"   Age: {age:.1f} | Income: ${income:.1f}K | Spending: {spending:.1f}")
    print(f"   Marketing Strategy: ", end="")
    
    if "High Spender" in profile:
        if "High Income" in profile:
            print("Premium products, loyalty programs")
        else:
            print("Aspirational marketing, payment plans")
    elif "Low Spender" in profile:
        if "Low Income" in profile:
            print("Budget products, discounts")
        else:
            print("Value propositions, bundle deals")
    else:
        print("Balanced approach, mid-range products")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

features = ['age', 'annual_income_(k$)', 'spending_score_(1-100)']
titles = ['Age Distribution', 'Income Distribution', 'Spending Score Distribution']

for i, (feature, title) in enumerate(zip(features, titles)):
    for cluster in range(optimal_k):
        cluster_data = df[df['Cluster_Age_Income_Spending'] == cluster][feature]
        axes[i].hist(cluster_data, alpha=0.5, label=f'Cluster {cluster}', bins=15)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(title)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()

model_filename = 'customer_segmentation_model.pkl'
scaler_filename = 'customer_scaler.pkl'

joblib.dump(kmeans2, model_filename)
joblib.dump(scaler2, scaler_filename)

print(f"\n✅ Model saved as {model_filename}")
print(f"✅ Scaler saved as {scaler_filename}")

prediction_script = """
import joblib
import numpy as np
import pandas as pd

model = joblib.load('customer_segmentation_model.pkl')
scaler = joblib.load('customer_scaler.pkl')

def predict_segment(age, income, spending_score):
    features = np.array([[age, income, spending_score]])
    features_scaled = scaler.transform(features)
    segment = model.predict(features_scaled)[0]
    return segment

def get_segment_description(segment):
    descriptions = {
        0: "Young Low Income Low Spender - Focus on budget products",
        1: "Middle-aged High Income High Spender - Premium customer",
        2: "Senior Middle Income Average Spender - Value seeker",
        3: "Young Middle Income High Spender - Aspirational buyer",
        4: "Middle-aged Middle Income Average Spender - Balanced shopper"
    }
    return descriptions.get(segment, "Unknown segment")

if __name__ == "__main__":
    test_customers = [
        (25, 35, 75),
        (45, 80, 20),
        (35, 50, 50),
        (60, 30, 85),
        (28, 90, 95)
    ]
    
    print("\\n📊 Customer Segment Predictions")
    print("="*50)
    for age, income, spending in test_customers:
        segment = predict_segment(age, income, spending)
        description = get_segment_description(segment)
        print(f"\\nAge: {age}, Income: ${income}K, Spending: {spending}")
        print(f"Segment {segment}: {description}")
"""

with open('segment_customers.py', 'w') as f:
    f.write(prediction_script)

print("✅ Prediction script 'segment_customers.py' created successfully!")

print("\n" + "="*60)
print("📋 PROJECT SUMMARY")
print("="*60)
print(f"""
✅ Dataset: Mall Customers Dataset
✅ Total Customers: {len(df)}
✅ Features Used: Age, Annual Income, Spending Score
✅ Optimal Clusters: {optimal_k}
✅ Model: K-means Clustering
✅ Silhouette Score: {silhouette_scores[optimal_k-2]:.3f}

Key Customer Segments Identified:
1. High-income high-spenders (Premium customers)
2. Young aspirational buyers
3. Budget-conscious shoppers
4. Average balanced customers
5. Senior value seekers

Files Created:
- task2.py (main script)
- segment_customers.py (prediction script)
- customer_segmentation_model.pkl (trained model)
- customer_scaler.pkl (scaler)
- elbow_method.png (optimal k visualization)
- income_spending_clusters.png (2D clusters)
- 3d_clusters.png (3D visualization)
- feature_distributions.png (feature analysis)
""")