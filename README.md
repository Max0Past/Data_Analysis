# IAD - Introduction to Data Analysis

A comprehensive repository containing five progressive laboratory projects focused on data analysis, machine learning, and advanced neural network techniques.

---

## Overview

### Summary

This repository contains a series of hands-on laboratory projects designed to provide a comprehensive introduction to data analysis and machine learning. The coursework progresses from fundamental data exploration and linear regression to advanced neural network architectures and clustering techniques. Each lab builds upon previously acquired knowledge, incorporating industry-standard tools and methodologies.

### Key Technologies & Concepts

- **Data Processing**: Python pandas, NumPy, CSV handling, data cleaning and preparation
- **Statistical Analysis**: Exploratory Data Analysis (EDA), descriptive statistics, data visualization
- **Machine Learning**: Linear Regression, Neural Networks, Classification, Clustering
- **Deep Learning Frameworks**: TensorFlow/Keras for sequential and custom neural architectures
- **Visualization Tools**: Matplotlib, Seaborn for exploratory and comparative visualization
- **Datasets**: Real-world datasets including avocado prices, Shanghai license plate prices, and Iris flowers
- **Programming Paradigm**: Object-oriented and functional approaches in Python
- **Model Evaluation**: MSE, MAE metrics, accuracy assessment, model performance comparison

---

## Laboratory Projects

### Lab 1: Fundamentals of Data Exploration

**Summary**
An introductory laboratory that establishes foundational knowledge in data exploration and basic statistical analysis. This lab serves as the entry point for understanding how to approach a raw dataset and extract meaningful insights.

**Tech Stack**

- Python 3
- Pandas for data manipulation
- NumPy for numerical operations
- Matplotlib for visualization

**Tech Challenge**
Understanding how to structure exploratory data analysis, handling missing values, and deriving initial insights from unprocessed data without proper methodology or visualization patterns.

**Solution**
Implemented systematic EDA workflow including:

- Dataset loading and initial inspection
- Statistical summary calculation
- Distribution analysis of key variables
- Correlation analysis between features
- Identification of missing values and outliers

**Impact**
Established a replicable framework for approaching any new dataset, providing a foundation for all subsequent laboratory projects and demonstrating the critical importance of exploratory analysis before modeling.

---

### Lab 2: Regression Analysis on Real-World Datasets

**Summary**
A practical application of linear regression models to two diverse real-world datasets: avocado prices and Shanghai license plate auction prices. This lab bridges the gap between theoretical understanding and practical implementation.

**Tech Stack**

- Python 3
- Pandas & NumPy for data processing
- Scikit-learn for regression modeling
- Matplotlib & Seaborn for visualization
- CSV data files

**Tech Challenge**
Handling multiple datasets with different characteristics, dealing with multicollinearity in features, feature selection, and model evaluation across diverse data distributions. Additionally, addressing data scaling and normalization issues that arise when comparing different measurement units and magnitudes.

**Solution**

- Conducted comprehensive EDA on both avocado and Shanghai datasets
- Implemented feature engineering and selection techniques
- Developed multiple linear regression models with performance comparison
- Applied data normalization and scaling where appropriate
- Evaluated models using MSE, MAE, and R² metrics
- Created comparative visualizations for model predictions vs. actual values

**Impact**
Demonstrated practical regression modeling skills and the ability to handle heterogeneous datasets. Provided insights into avocado market dynamics and Shanghai license plate pricing mechanisms, showing how machine learning reveals patterns in real-world economic data.

---

### Lab 3: Neural Networks for Regression

**Summary**
An exploration of artificial neural networks applied to the same datasets from Lab 2, comparing the performance and characteristics of traditional regression versus deep learning approaches for regression tasks.

**Tech Stack**

- Python 3
- TensorFlow/Keras for neural network development
- Pandas & NumPy for data preprocessing
- Scikit-learn for data splitting and metrics
- Matplotlib for training visualization

**Tech Challenge**
Designing appropriate network architectures, selecting activation functions, optimizing hyperparameters (learning rate, batch size, epochs), handling overfitting, and comparing neural network performance against classical regression methods. Additionally, implementing proper data normalization for deep learning models.

**Solution**

- Designed multi-layer neural network architectures (input, hidden, output layers)
- Implemented sequential models with various activation functions (ReLU, sigmoid)
- Applied proper data normalization techniques
- Implemented early stopping to prevent overfitting
- Visualized training history (loss curves)
- Compared neural network performance metrics with traditional regression models
- Optimized hyperparameters through experimentation

**Impact**
Demonstrated that neural networks can provide competitive or superior performance to traditional methods when properly tuned. Highlighted the trade-off between model complexity and interpretability, and established best practices for neural network training and evaluation.

---

### Lab 4: Classification and Iris Dataset Analysis

**Summary**
An introduction to classification problems using the classic Iris dataset and synthetic datasets. This lab transitions from regression to classification and demonstrates both traditional ML and neural network approaches for categorical prediction.

**Tech Stack**

- Python 3
- Scikit-learn for classical ML algorithms
- TensorFlow/Keras for neural classification networks
- Pandas & NumPy for data handling
- Matplotlib & Seaborn for visualization
- CSV and Python data modules

**Tech Challenge**
Implementing classification models, choosing appropriate network architectures for multi-class problems, handling balanced vs. unbalanced datasets, and generating synthetic data for controlled experimentation. Understanding the differences between regression and classification paradigms.

**Solution**

- Loaded and analyzed the Iris dataset with detailed exploratory analysis
- Implemented neural networks for multi-class classification (3 classes)
- Generated synthetic datasets (moons) for more complex classification scenarios
- Applied data preprocessing and normalization
- Compared classification metrics (accuracy, precision, recall)
- Visualized decision boundaries and model predictions
- Trained models on both real and synthetic datasets

**Impact**
Expanded the toolkit to classification problems and demonstrated how neural networks handle multi-class prediction. Provided practical experience with synthetic datasets and established techniques for evaluating classification model performance.

---

### Lab 5: Clustering with Make Circles Dataset

**Summary**
An advanced laboratory focusing on unsupervised learning through clustering techniques. This lab addresses non-linearly separable data and explores clustering algorithms beyond simple distance-based approaches.

**Tech Stack**

- Python 3
- Scikit-learn for clustering algorithms
- TensorFlow/Keras for neural network clustering approaches
- NumPy & Pandas for data manipulation
- Matplotlib for visualization

**Tech Challenge**
Handling non-linearly separable data that cannot be adequately partitioned using traditional clustering methods. Understanding the limitations of K-means clustering and exploring alternative approaches such as spectral clustering or density-based methods.

**Solution**

- Generated the make_circles synthetic dataset
- Attempted traditional clustering approaches and identified limitations
- Implemented advanced clustering techniques suitable for non-linear separability
- Applied data normalization for consistent clustering
- Visualized cluster assignments and decision boundaries
- Compared different clustering algorithm performances
- Analyzed cluster quality and cohesion metrics

**Impact**
Demonstrated the importance of choosing appropriate algorithms for specific data distributions. Highlighted that algorithm selection must match data characteristics, and that neural networks and advanced clustering methods are necessary when traditional approaches fail.

---

## Repository Structure

```
IAD/
├── README.md                          # Project documentation
├── lab_1/
│   └── Lab_notebook.ipynb            # Fundamentals of data exploration
├── lab_2/
│   ├── avocado_prices_code.ipynb     # Avocado regression analysis
│   ├── Shanghai_code.ipynb           # Shanghai license plates regression
│   └── data/
│       ├── avocado.csv               # Raw avocado dataset
│       ├── avocado_prepared.csv      # Processed avocado dataset
│       └── Shanghai license plate price - Sheet3.csv  # Shanghai pricing data
├── lab_3/
│   ├── avocado_prices_code_NN.ipynb  # Avocado neural network regression
│   ├── Shanghai_code_NN.ipynb        # Shanghai neural network regression
│   └── data/
│       ├── avocado.csv
│       └── avocado_prepared.csv
├── lab_4/
│   ├── load_iris_code.ipynb          # Iris classification with neural networks
│   ├── make_moons_code.ipynb         # Synthetic data classification
│   └── data/
│       ├── 1load_data.py             # Data loading utilities
│       ├── iris_dataset.csv          # Iris flower dataset
│       └── moons_dataset.csv         # Synthetic moons dataset
├── lab_5/
│   └── make_circles.ipynb            # Clustering with synthetic circles data
└── Захист практикуму №1/
    └── check_positive_array_cases.ipynb  # Test case validation

```

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Required packages: pandas, numpy, scikit-learn, tensorflow, keras, matplotlib, seaborn

### Installation

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn jupyter
```

### Running the Labs

Navigate to the desired lab directory and open the notebook:

```bash
jupyter notebook lab_1/Lab_notebook.ipynb
```

---

## Learning Outcomes

Upon completion of all laboratories, participants will have:

✓ Mastered exploratory data analysis techniques
✓ Implemented both classical regression and neural network regression models
✓ Applied classification algorithms to real and synthetic datasets
✓ Developed clustering solutions for complex, non-linearly separable data
✓ Gained practical experience with TensorFlow/Keras framework
✓ Learned to evaluate and compare model performance
✓ Understood the progression from simple to complex machine learning architectures
✓ Developed critical thinking about algorithm selection based on data characteristics

---

## Author

Implementation & Documentation by **Maksym Pastushenko, Zlata Plakhtii , Felix Rovmanov**

Created as part of Introduction to Data Analysis coursework developed by **Nadezhda Nedashkovskaya** at National Technical University of Ukraine "Igor Sikorsky Kyiv Polytechnic Institute" (NTUU "KPI"), Institute of Applied Systems Analysis, Department of Mathematical Methods of Systems Analysis

---
