# SVM for Linear and Non-Linear Classification

I built this project to explore how Support Vector Machines (SVMs) can be used for both linear and non-linear classification problems. I used the Breast Cancer dataset and focused on visualizing how the decision boundaries change with different kernel functions.

## 🔍 What I Did

- Loaded and prepared the Breast Cancer dataset (binary classification).
- Trained an SVM model using two different kernels:
  - **Linear Kernel**
  - **RBF (Radial Basis Function) Kernel**
- Visualized the decision boundaries in 2D to better understand how each kernel behaves.
- Tuned hyperparameters like `C` and `gamma` to see their effect on the model.
- Used cross-validation to evaluate the model’s performance more reliably.

## 📊 Tools I Used

- Python
- Scikit-learn
- Matplotlib
- NumPy

## 📌 How to Run

1. Make sure you have Python installed.
2. Install required libraries if not already installed:
   ```bash
   pip install numpy scikit-learn matplotlib
