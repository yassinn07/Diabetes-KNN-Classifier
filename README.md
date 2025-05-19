# 🔍 Custom K-Nearest Neighbors (KNN) Classifier on Diabetes Dataset

This project implements a simple KNN classifier **from scratch** (without using any built-in KNN libraries) on the diabetes dataset (`diabetes.csv`). The goal is to classify diabetes presence using Euclidean distance and a custom tie-breaking mechanism based on distance-weighted voting.

---

## 🎯 Objective

- Perform classification over multiple iterations of different K values (e.g., K=2, 3, 4, ...).
- Use **Euclidean distance** to measure similarity between instances.
- Implement a **distance-weighted voting** scheme to break ties by giving higher influence to closer neighbors.
- Split data into 70% training and 30% testing.

---

## 🧹 Data Preprocessing

- Normalize each feature separately for training and testing sets.
- Normalization methods available:
  - **Log Transformation** or
  - **Min-Max Scaling** (recommended)
- Normalization is applied independently on training and test data.

---

## ⚙️ Implementation Details

### 1. Data Splitting
- Randomly split the dataset: 70% training, 30% testing.

### 2. Distance Calculation
- Compute Euclidean distance between test instance and all training instances.

### 3. KNN Classification
- For each test instance, find the K nearest neighbors.
- Collect votes from neighbors’ classes.
- If there is a tie between classes:
  - Use **distance-weighted voting** to break the tie.
  - Closer neighbors have higher weights (e.g., weight = 1 / distance).
  
### 4. Iterations
- Run the classification for multiple K values (e.g., K=2 to K=6).
- For each K, run 5 iterations with different train-test splits (random states).

---

## 📊 Output

For each iteration and each K value, print:
- K value
- Number of correctly classified test instances
- Total number of test instances
- Accuracy for that iteration

At the end, print the average accuracy across all iterations and K values.
