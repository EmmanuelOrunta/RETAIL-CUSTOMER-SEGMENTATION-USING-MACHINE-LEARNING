# 🛍️ Retail Customer Segmentation (Australian Market)

## 📌 Project Overview

This project focuses on segmenting retail customers into meaningful groups based on their purchasing behaviour. The goal is to help businesses better understand their customers and make smarter marketing and business decisions.
Using a combination of **customer transaction data** and **Australian retail trends**, this project applies **machine learning techniques** to identify key customer segments such as:

- High-value customers 💰  
- Loyal customers 🔁  
- At-risk customers ⚠️  
- Low-value customers 📉

This project simulates a real-world scenario where businesses use data to personalise customer experiences and improve revenue.

---  
## 🎯 Business Problem

Retail businesses often struggle to answer questions like:
- Who are my most valuable customers?
- Which customers are likely to stop buying?
- How can I target different customers effectively?

This project solves these problems by grouping customers into segments based on their behaviour.

---
## 📊 Datasets Used (Hybrid Approach)

This project uses two datasets to simulate a realistic business environment:
---

### 🟢 1. Online Retail Dataset (Primary Dataset)
- Source: Kaggle  
- Contains detailed transactional data including:
  - Customer ID  
  - Purchase history  
  - Quantity and price  
  - Transaction dates

👉 This dataset is used for:  
- Customer behaviour analysis  
- RFM feature creation  
- Machine learning (clustering)  

---
### 🟡 2. Australian Retail Dataset (Context Dataset)
- Source: Kaggle 
- Contains aggregated retail sales data in Australia

👉 This dataset is used for:
- Understanding market trends
- Adding local Australian business context 
- Supporting storytelling and insights

---
## 🧠 Key Concepts Used
### 🔹 1. RFM Analysis
RFM stands for:
- **Recency (R)** → How recently a customer made a purchase
- **Frequency (F)** → How often they purchase
- **Monetary (M)** → How much money they spend 

👉 This helps convert raw transaction data into meaningful customer profiles.

---
### 🔹 2. Feature Scaling

Feature scaling ensures all variables contribute equally to the model.

Without scaling:
- Large values (e.g., monetary) dominate the model

With scaling:
- All features are treated fairly

---
### 🔹 3. Unsupervised Learning

This project uses **unsupervised learning**, meaning:
- There are no predefined labels  
- The model discovers patterns on its own

---
### 🔹 4. Clustering

Clustering is the process of grouping similar customers together.

👉 Customers in the same group behave similarly.

---
### 🔹 5. K-Means Algorithm

K-Means is a clustering algorithm that:

1. Chooses a number of clusters (K)
2. Assigns customers to the nearest group
3. Adjusts group centers
4. Repeats until stable

---
### 🔹 6. Elbow Method
The Elbow Method is used to determine the optimal number of clusters.

👉 It finds the point where adding more clusters no longer significantly improves the model.

---
## ⚙️ Project Workflow
### 1. Data Loading
- Load both datasets using pandas  

### 2. Data Cleaning
- Remove missing values
- Remove invalid transactions
- Convert date formats

### 3. Feature Engineering
- Create RFM metrics for each customer  

### 4. Data Scaling
- Normalize features to ensure fairness

### 5. Model Building
- Apply K-Means clustering
