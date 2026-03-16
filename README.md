# PRODIGY_ML_02 - Customer Segmentation using K-Means Clustering

## Task Description
Create a K-means clustering algorithm to group customers of a retail store based on their purchase history.

## Dataset
- **Source**: Mall Customers Dataset
- **Size**: 200 customers
- **Features**: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)

## Objectives
- Segment customers into distinct groups
- Identify customer patterns and behaviors
- Provide marketing strategies for each segment

## Methodology
1. **Data Preprocessing**: Cleaned column names, checked for missing values
2. **Feature Selection**: Age, Annual Income, Spending Score
3. **Optimal K Selection**: Used Elbow Method and Silhouette Score
4. **K-Means Clustering**: Applied with k=5 clusters
5. **Segment Analysis**: Profiled each customer group

## Results

### Optimal Number of Clusters: 5
**Silhouette Score**: 0.553

### Customer Segments Identified:

| Cluster | Profile | Age | Income | Spending | Marketing Strategy |
|---------|---------|-----|--------|----------|-------------------|
| 0 | Young Low Income Low Spender | 25.5 | $26.3K | 20.5 | Budget products, discounts |
| 1 | Middle-aged High Income High Spender | 41.1 | $86.5K | 82.1 | Premium products, loyalty programs |
| 2 | Senior Middle Income Average Spender | 55.8 | $48.5K | 45.2 | Value propositions, bundle deals |
| 3 | Young Middle Income High Spender | 32.1 | $58.3K | 75.8 | Aspirational marketing, payment plans |
| 4 | Middle-aged Middle Income Average Spender | 42.5 | $54.2K | 48.3 | Balanced approach, mid-range products |

## Files Included
- `task2.py` - Main Python script for clustering
- `segment_customers.py` - Prediction script for new customers
- `customer_segmentation_model.pkl` - Trained K-means model
- `customer_scaler.pkl` - StandardScaler for feature normalization
- `Mall_Customers.csv` - Dataset
- `elbow_method.png` - Optimal k visualization
- `income_spending_clusters.png` - 2D cluster visualization
- `3d_clusters.png` - 3D cluster visualization
- `feature_distributions.png` - Feature analysis by cluster

## How to Run

1. **Install required packages:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
