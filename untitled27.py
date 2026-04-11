
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
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# CREATE RFM FEATURES
reference_date = df['InvoiceDate'].max()

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# STEP 5: SCALE FEATURES
# ================================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
