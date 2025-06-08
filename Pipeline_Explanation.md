

# Final Project Pipeline Description

This document outlines the data preprocessing pipeline constructed for the final project. The goal of this project is to predict the likelihood of 5G network rollout (`5G_Rollout`) based on a country's digital and economic indicators.

To ensure that the data is prepared correctly and the modeling process is both robust and reproducible, I have constructed a `Pipeline` from scikit-learn, using the custom transformers provided in the course library. This pipeline automates all the necessary feature transformation steps before the data is fed into a machine learning model.

---

## Pipeline Steps

The pipeline sequentially applies a series of transformations to the raw data. Each step is designed to address a specific preprocessing need, such as handling categorical data or scaling numeric features. The `verbose=True` parameter ensures that the completion of each step is printed during execution.

### 1. Categorical Feature Encoding

* **Step Name**: `map_urban_rural`
* **Transformer**: `CustomMappingTransformer`
* **Purpose**: Machine learning models require numerical input. This first step converts the `Urban_Rural` column, which contains text values ('Urban', 'Rural'), into a binary numerical format.
    * `'Urban'` is mapped to `1`.
    * `'Rural'` is mapped to `0`.
    This process is directly comparable to the 'Gender' mapping in the Titanic example.

### 2. Outlier Handling

* **Purpose**: Numeric features can often contain outliers (extremely high or low values) that can negatively skew the training of a model. The following steps identify and clip these outliers in key numeric columns using Tukey's outer fence method (values outside Q1 - 3\*IQR and Q3 + 3\*IQR).
* **Steps**:
    * `tukey_gdp`: Applies outlier clipping to the `GDP_Per_Capita (USD)` column.
    * `tukey_speed`: Applies outlier clipping to the `Broadband_Speed (Mbps)` column.
    * `tukey_investment`: Applies outlier clipping to the `Digital_Investment (M USD)` column.

### 3. Numeric Feature Scaling

* **Purpose**: After handling outliers, the numeric features are scaled to bring them to a similar range. This prevents features with naturally large scales (like GDP) from dominating the learning process over features with smaller scales (like broadband speed).
* **Transformer**: `CustomRobustTransformer`
* **Method**: This transformer uses a method that is robust to the presence of remaining outliers. It scales the data by subtracting the median and dividing by the Interquartile Range (IQR). This is applied to the same columns that underwent outlier handling.
* **Steps**:
    * `scale_gdp`: Scales the `GDP_Per_Capita (USD)` column.
    * `scale_speed`: Scales the `Broadband_Speed (Mbps)` column.
    * `scale_investment`: Scales the `Digital_Investment (M USD)` column.

### Omitted Step: Missing Value Imputation

The `titanic_transformer` example included a final step using `CustomKNNTransformer` to impute missing values. After analyzing this project's dataset with `.info()`, it was determined that there are **no missing values**. Therefore, the imputation step is intentionally and correctly omitted from this pipeline, demonstrating a preprocessing workflow tailored specifically to the dataset's characteristics.
