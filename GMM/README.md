# Project: Wine Quality Classification and Clustering Using GMM and LDA

## 1. Overview
This project focuses on classifying and clustering wine samples using two different datasets: **white wine** and **red wine**. The main objectives are:
- To explore the differences between red and white wines.
- To leverage machine learning models for clustering.
- To use **Linear Discriminant Analysis (LDA)** for dimensionality reduction to improve clustering performance.

## 2. Models Used

### 2.1 Gaussian Mixture Model (GMM)
The **Gaussian Mixture Model** is a probabilistic model used for clustering. It assumes that the data is generated from a mixture of several Gaussian distributions.

#### **Key Features of GMM**:
- **Soft Clustering**: Each data point is assigned a probability of belonging to each cluster, rather than being assigned to a single cluster.
- **Flexibility**: GMM can model clusters with different shapes, sizes, and orientations.
- **Parameters**:
  - `n_components`: Number of Gaussian components (clusters).
  - `covariance_type`: Covariance type of the Gaussian distributions (`full`, `tied`, `diag`, `spherical`).

In this project, GMM was used to identify clusters in the wine dataset. The model was trained on features after applying dimensionality reduction.

---

### 2.2 Linear Discriminant Analysis (LDA)
**LDA** is a supervised dimensionality reduction technique. It projects high-dimensional data onto a lower-dimensional space while preserving the separability of different classes.

#### **LDA's Core Principles**:
1. **Maximizing Class Separation**:
   - LDA seeks a projection direction that maximizes the distance between the means of different classes (class separation).
2. **Minimizing Intra-class Variability**:
   - It ensures that samples from the same class stay close together after projection.

#### **Mathematical Explanation**:
- LDA calculates two scatter matrices:
  - **Within-class scatter matrix \(S_W\)**:
    \[
    S_W = \sum_{c=1}^C \sum_{x_i \in c} (x_i - \mu_c)(x_i - \mu_c)^T
    \]
    Measures the compactness of samples within the same class.
  - **Between-class scatter matrix \(S_B\)**:
    \[
    S_B = \sum_{c=1}^C n_c (\mu_c - \mu)(\mu_c - \mu)^T
    \]
    Measures the separation between different class centers.
  
- The optimal projection matrix \(W\) is obtained by maximizing the following ratio:
  \[
  W = \arg \max \frac{|W^T S_B W|}{|W^T S_W W|}
  \]
  This ensures the maximum separation between classes after projection.

#### **Purpose of LDA in This Project**:
- The wine dataset contains 12 features, which are high-dimensional.
- **LDA** reduces the data to 1 dimension (since we have 2 classes: red and white wine), which simplifies clustering.
- Improved clustering performance was observed after applying LDA.

---

## 3. Workflow

1. **Data Preprocessing**:
   - Load white and red wine datasets.
   - Combine features from both datasets and assign labels (1 for white wine, 2 for red wine).
   - Standardize the features using **StandardScaler**.

2. **Dimensionality Reduction**:
   - Apply **LDA** to reduce the dimensionality from 12 to 1.

3. **Model Training**:
   - Train a **GMM** on the LDA-reduced data.

4. **Evaluation**:
   - Use **Silhouette Score** to evaluate the compactness and separation of clusters.
   - Use **Adjusted Rand Index (ARI)** to measure the agreement between the predicted clusters and the actual labels.

---

## 4. Results

After applying LDA and GMM, the following performance metrics were observed:
- **Silhouette Score**: 0.8155
  - Indicates a strong clustering structure with well-separated clusters.
- **Adjusted Rand Index (ARI)**: 0.9672
  - Demonstrates excellent alignment between the predicted clusters and the actual labels.

These results confirm that LDA significantly improved clustering performance by reducing dimensionality while retaining the essential characteristics of the data.

---

## 5. How to Run the Project

1. Ensure you have the required Python packages installed:
   ```bash
   pip install -r requirements.txt
