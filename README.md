
# 📊 **Customer Segmentation using Machine Learning (K-Means + PCA + Pipeline)**

A complete **industry-style** customer segmentation project using **Scikit-Learn Pipelines**, **ColumnTransformer**, **PCA**, and **KMeans** applied on the **Mall Customers dataset**.

This project demonstrates how businesses cluster customers into different groups for **marketing**, **personalization**, and **business insights**.

---

## 🚀 **Project Overview**

Customer segmentation helps businesses understand customer groups based on behavior, spending, and demographics.
This project builds an **end-to-end ML pipeline** that:

✔ Preprocesses numerical & categorical data
✔ Scales features
✔ Applies PCA for dimensionality reduction
✔ Performs KMeans clustering
✔ Generates cluster insights
✔ Visualizes results
✔ Saves the full ML pipeline using Joblib

---

## 📁 **Dataset**

**Mall Customer Segmentation Dataset**

* **Features:** Age, Gender, Annual Income, Spending Score
* **Use Case:** Cluster customers into meaningful segments

Dataset Link:
👉 [https://www.kaggle.com/datasets/shwetabh123/mall-customers](https://www.kaggle.com/datasets/shwetabh123/mall-customers)

---

## 🧠 **Machine Learning Pipeline**

The project uses a **clean, modular, production-level pipeline**:

### 🔧 **Preprocessing**

* `StandardScaler` → scales numeric features
* `OneHotEncoder` → encodes Gender
* `ColumnTransformer` → combines preprocessing

### 🎯 **Modeling**

* `PCA(n_components=2)` → reduces dimensionality
* `KMeans(n_clusters=5)` → clusters customers

### 🏗 **Full Pipeline**

All steps combined using:

```python
Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=2)),
    ('kmeans', KMeans(n_clusters=5))
])
```

---

## 📦 **Installation**

```bash
git clone https://github.com/yourusername/customer-segmentation-ml.git
cd customer-segmentation-ml
pip install -r requirements.txt
```

---

## 🐍 **Run the Project**

Run the Python script or notebook:

```bash
python customer_segmentation.py
```

Or open the Jupyter notebook:

```bash
jupyter notebook
```

---

## 📉 **Cluster Visualization (PCA 2D Plot)**

The clusters are visualized using PCA to project high-dimensional data into 2D.

Example output:

```
Cluster 0: High income, high spending  
Cluster 1: Low income, low spending  
Cluster 2: High income, low spending  
Cluster 3: Young, moderate spending  
Cluster 4: Older, moderate spending  
```

---

## 📊 **Cluster Profiling**

Group-level insights:

```python
df.groupby("cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()
```

This helps businesses understand each customer group clearly.

---

## 💾 **Save & Load the Model**

### Save:

```python
joblib.dump(pipeline, "customer_segmentation_pipeline.pkl")
```

### Load:

```python
pipeline = joblib.load("customer_segmentation_pipeline.pkl")
```

---

## 🛠 **Tech Stack**

* Python
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib

---

## 🧩 **Project Structure**

```
📁 customer-segmentation-ml/
│── 📄 customer_segmentation.py
│── 📄 README.md
│── 📄 requirements.txt
│── 📄 customer_segmentation_pipeline.pkl
│── 📁 data/
│      └── Mall_Customers.csv
│── 📁 visuals/
│      └── cluster_plot.png
```

---

## 📌 **Key Learnings**

✔ How to build a complete ML pipeline
✔ Feature scaling & encoding
✔ Dimensionality reduction with PCA
✔ Clustering with KMeans
✔ Silhouette score evaluation
✔ Visualizing clusters
✔ Saving ML pipelines for production use

---

## ⭐ **Future Improvements**

* Add auto-selection of best number of clusters (Elbow + Silhouette)
* Deploy model using FastAPI
* Add dashboard using Streamlit or Power BI

---

## 🤝 **Contributions**

Pull requests are welcome!
If you find a bug or want a new feature, feel free to open an issue.

---

## 📬 **Contact**

**Muhammad Tanveer**

