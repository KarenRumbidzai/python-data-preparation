# Retail Customer Churn: Data Exploration & Preparation

> **Academic assignment.** KBIDA421 Assignment 1, BSc Business Intelligence and Data Analytics, Women's University in Africa (WUA).

A Python script that takes a raw retail customer dataset and turns it into a clean, modelling-ready file for churn prediction.

## What it does

1. **Describes the dataset**: observations, variables, data types
2. **Defines the target**: `Churn` (binary, Yes/No)
3. **Assesses data quality**: missing values, duplicate rows and Customer IDs, invalid values, extremes
4. **Summarises the data**: descriptive statistics with skewness
5. **Visualises**: age histogram, income/spending boxplots, churn rate by region and by campaign response
6. **Checks relationships**: income vs spending, and support calls, purchases and website visits vs churn
7. **Cleans**: removes duplicate IDs, median-imputes income and spending, mode-imputes payment method
8. **Handles outliers**: IQR rule with winsorising (capped, not deleted)
9. **Encodes**: binary encoding for campaign response and churn, one-hot for payment method and region
10. **Engineers features**: `Spend_to_Income_Ratio`, `Avg_Spend_per_Purchase`, `Visit_to_Purchase_Ratio`, `High_Support_Calls_Flag`
11. **Documents**: builds a final data dictionary

## Requirements

- Python 3.8+
- `pandas`, `numpy`, `matplotlib`

```bash
pip install pandas numpy matplotlib
```

## Usage

Place `KBIDA421_Assignment_1_Retail_Customer_Dataset.csv` in the same folder as the script, then run:

```bash
python data-preparataion.py
```

## Outputs

| File                                          | Description                                               |
| --------------------------------------------- | --------------------------------------------------------- |
| `charts/*.png`                                | Four exploratory charts                                   |
| `retail_customer_dataset_modelling_ready.csv` | Cleaned, encoded dataset with engineered features         |
| `data_dictionary.csv`                         | Description of every variable and the preparation applied |

## Author

Karen Rumbidzai Mufandaedza
W240155
