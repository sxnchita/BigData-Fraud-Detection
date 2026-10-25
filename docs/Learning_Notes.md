# 📘 Big Data Analytics - Learning Notes

**Student:** Sanchita Senapati

**Semester:** 5

**Subject:** Big Data Analytics

**Project Title (Working):**
Scalable Big Data Analytics Framework for Financial Transaction Fraud Detection Using Apache Spark

---

# Day 1 - Environment Setup & Dataset Exploration

## Objective

Learn how to set up the Python environment and load a real-world fraud detection dataset.

## What I Learned

- Installed Python 3.13.
- Configured Visual Studio Code.
- Installed the Python and Jupyter extensions.
- Learned what a Jupyter Notebook is.
- Imported the Pandas library.
- Loaded a CSV dataset using `pd.read_csv()`.
- Displayed the first five rows using `df.head()`.
- Checked the dataset size using `df.shape`.
- Viewed dataset information using `df.info()`.

## Dataset Details

- Dataset: Credit Card Fraud Detection
- Source: Kaggle (Machine Learning Group - ULB)
- Total Transactions: **284,807**
- Total Columns: **31**

## Key Concepts

- A **DataFrame** is a table-like structure used in Pandas.
- One **row** represents one **credit card transaction**.
- The **target column** is `Class`.
- `Class = 0` → Legitimate Transaction
- `Class = 1` → Fraudulent Transaction

## Commands Learned

```python
import pandas as pd
```

```python
df = pd.read_csv("../data/raw/creditcard.csv")
```

```python
df.head()
```

```python
df.shape
```

```python
df.info()
```

## Reflection

Today I successfully loaded my first real-world dataset into Python. I learned how to inspect the dataset and understand its structure before beginning any analysis.
# Day 2 - Data Quality Check

## Objective

To assess the quality of the dataset before performing analysis and model building.

---

## Concept Learned

### Missing Values

Missing values are blank or unavailable entries in a dataset. They can reduce the accuracy of analysis and machine learning models if not handled properly.

---

## Command Used

```python
df.isnull().sum()
```

### Purpose

Counts the number of missing values in each column.

---

## Observation

Every column contains **0 missing values**.

Therefore, the dataset does not require missing value treatment.

---

## Viva Question

**Q:** Why do we check for missing values?

**A:** Missing values can affect data analysis and machine learning models. Checking them helps ensure the dataset is complete and reliable.