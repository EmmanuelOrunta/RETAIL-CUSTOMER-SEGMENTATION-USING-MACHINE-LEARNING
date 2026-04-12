
# RETAIL CUSTOMER SEGMENTATION USING MACHINE LEARNING (HYBRID DATA APPROACH)

#  IMPORT LIBRARIES
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

# ================================
# STEP 6: ELBOW METHOD
# ================================
inertia = []

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
