
# RETAIL CUSTOMER SEGMENTATION USING MACHINE LEARNING (HYBRID DATA APPROACH)

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# LOAD DATASETS
# Dataset 1: Online Retail (Customer-level data)
df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')
# Dataset 2: Australian Retail (Context dataset)
aus_df = pd.read_csv("australian_retail.csv")


# DATA CLEANING OF ONLINE RETAIL DATA

df = df.dropna(subset=['CustomerID'])  # Remove missing customers
df = df[df['Quantity'] > 0]            # Remove returns/negative sales
# Convert date
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
# Create revenue column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice'] # Monetary 

# CREATE RFM FEATURES
reference_date = df['InvoiceDate'].max()

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days, # Recency
    'InvoiceNo': 'count', # Frequency
    'TotalPrice': 'sum' 
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# SCALE FEATURES
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm) # It puts all rfm variables on the same level


# ELBOW METHOD - a method of choosing clusters

inertia = [] # how tight the clusters are

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 10), inertia)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# APPLY K-MEANS - A method used for grouping things in similar clusters

kmeans = KMeans(n_clusters=4, random_state=42) # 4 clusters
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)


# LABEL SEGMENTS
def label_cluster(row):
    if row['Cluster'] == 0:
        return 'High Value'
    elif row['Cluster'] == 1:
        return 'At Risk'
    elif row['Cluster'] == 2:
        return 'Loyal'
    else:
        return 'Low Value'
rfm['Segment'] = rfm.apply(label_cluster, axis=1)


# CLUSTER SUMMARY

cluster_summary = rfm.groupby('Segment').mean() # Shows average behaviour per segment
# Example insight:
# High Value → high money, high frequency
# At Risk → high recency (haven’t bought recently)

print("\nCustomer Segment Summary:")
print(cluster_summary)


# AUSTRALIAN DATA ANALYSIS (CONTEXT)

# Convert date if exists
if 'Date' in aus_df.columns:
    aus_df['Date'] = pd.to_datetime(aus_df['Date'])

# Example: Monthly sales trend
if 'Sales' in aus_df.columns:
    monthly_sales = aus_df.groupby(aus_df['Date'].dt.to_period('M'))['Sales'].sum()
    
    plt.figure()
    monthly_sales.plot()
    plt.title("Australian Retail Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.show()

# MERGE INSIGHTS (SIMULATION)
# Add a simulated "Region" to customers to align with Australian context
regions = ['NSW', 'VIC', 'QLD', 'WA']
rfm['Region'] = np.random.choice(regions, size=len(rfm))
# Analyze segments by region
region_analysis = rfm.groupby(['Region', 'Segment']).size().unstack()
print("\nSegment Distribution by Region:")
print(region_analysis)

# EXPORT RESULTS
rfm.to_csv("customer_segments_hybrid.csv")

print("\nProject Completed Successfully!")
