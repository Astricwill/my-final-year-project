# Student Lifestyle Preprocessing and Modeling

This project processes and analyzes a student lifestyle dataset from Topfaith University. It focuses on cleaning the data, transforming categorical values, creating a lifestyle score, and comparing machine learning models for lifestyle classification.

## What this project includes

- `preprocessing.ipynb`: A Jupyter notebook that performs the full preprocessing workflow and model evaluation.
- `data/Lifestyle Dataset of Students at Topfaith University.csv`: The original raw dataset used for analysis.
- `Cleaned_dataset.csv`: The cleaned dataset produced after preprocessing.
- `Graph.pdf`: A generated graph report or visual summary (if available).

## Notebook workflow

1. Import the necessary Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, and scikit-learn.
2. Load the original dataset from `data/Lifestyle Dataset of Students at Topfaith University.csv`.
3. Remove irrelevant columns such as `Timestamp`.
4. Rename columns to more readable feature names.
5. Encode categorical columns using label encoding.
6. Visualize distributions and correlations with boxplots and a heatmap.
7. Create a lifestyle scoring column based on:
   - Sleep hours
   - Exercise frequency
   - Screen time
   - Diet quality
   - Stress frequency
8. Classify students into lifestyle categories:
   - `0`: Unhealthy
   - `1`: Moderately Healthy
   - `2`: Healthy
9. Save the cleaned dataset to `Cleaned_dataset.csv`.
10. Generate visualizations showing how lifestyle relates to key features.
11. Train and compare three classification models:
- Logistic Regression
- Decision Tree
- Random Forest

12. Evaluate models using accuracy, precision, recall, and F1 score.

## How to use

1. Open `preprocessing.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the notebook cells from top to bottom.
3. Review the visualizations and model comparison output.

## Purpose

The goal is to transform the student lifestyle dataset into a usable form for analysis and classification, then compare machine learning models for predicting lifestyle health categories.

## Notes

- This project is primarily exploratory and for academic use.
- The classification labels are built from a simple rule-based scoring system, not from external ground truth.
- Further improvements could include better feature engineering, cross-validation, and using more advanced classification methods.