
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
