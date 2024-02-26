# Dunnhumby Coupon Redemption

## What it does:
A classification model that predicts whether or not a given customer in a group of frequent shoppers will redeem a coupon.
Data sourced from [Kaggle](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)

This Capstone Project was built as part of the academic requirement of the PGP-DSE program at Great Learning Hyderabad.

Sometimes GitHub is unable to Preview Code Blocks for Jupyter Notebooks. If this happens, you can just view my [Notebook](https://nbviewer.org/github/galahad38/dunnhumby-coupon-redemption/blob/main/dunnhumby-coupon-redemption.ipynb).

## How to build it yourself:

1. Install [Python](https://www.python.org/downloads/).
2. Install non-standard Python libraries:
     launch command prompt and run this command:
     ```console
     C:\Windows\system32\ pip install ipykernal, jupyterlab, notebook, numpy, pandas, matplotlib, seaborn, scikit-learn, statsmodels
     ```
3. Download the [Jupyter Notebook](https://github.com/galahad38/dunnhumby-coupon-redemption/blob/main/dunnhumby-coupon-redemption.ipynb).
4. Download the [Dataset](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey) from Kaggle.
5. Launch Jupyter Notebook from the Start Menu, and navigate to the folder containing the dataset and Jupyter Notebook you just downloaded.
6. Extract the csv files into a folder named 'archive'. Ensure that the Jupyter Notebook is in the same folder as 'archive'.
7. Go to Cell -> Run All.
8. Profit!

## How to interpret it:

1) The EDA portion answers some important questions that arise regarding the data.
2) The Data was not present in a monolithic form, several features were created using the different tables and joined to create the base DataFrame.
3) Several Classification Models were built, the best of which had a Macro Average F1-Score of 0.7

## My instance and the insights derived:

### Data Collection:
8 .csv files were downloaded off of [Kaggle](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)
Since we didn't have a monolithic DataFrame, the HH_demographic DataFrame was taken and several numerical features were created and added to the DataFrame in order to build the models.

### Feature Engineering:

First things first, the HH_demographic DataFrame didn't contain the Target Variable (Redeemed).
If the household_key (unique identifier) was present in the coupon_redempt DataFrame, the  the Target Variable was assigned the value 1, and if not, then 0.

Next, the HH_demographic DataFrame contained categorical variables that were static descriptors of the customers/households, such as income bracket (INCOME_DESC) , and household compostion (HH_COMP_DESC)
I wanted to add numerical features that would be "dynamic" descriptors of the customers'/households' shopping habbits.
I believed the following features would contribute useful information to the model:

1) Number of Campaigns a Household was targeted for
2) Number of Distinct Coupons a Household Redeemed
3) Coupon Success Ratio - Ratio of Coupons Redeemed to Coupons Received
4) Average Number of Items a Household Purchases Per Visit
5) Number of Visits a Household Pays to the Retail Store
6) Average Amount Spent by a Household Per Visit

### Data Cleaning:
After the features were created, several defects remained:
1) Null Values in the Numerical Features created were imputed with zero, since the value didn't exist for said household_key (unique identifier)
2) Outliers were treated using Winsorization (Capping)
3) The Data was split into Train and Test sets.
4) The numerical variables were Scaled and the categorical variables were encoded for both Train and Test sets.
5) The categorical variables were binned better than they were initially.

### Exploratory Data Analysis:

The following questions were answered:
1) On average, how long did each type of campaign run for?
![Untitled](https://github.com/galahad38/dunnhumby-coupon-redemption/assets/19240929/81d76193-b339-4a44-8577-5f1c421a34a8)
On average, TypeB campaigns run the shortest, for 37.6 days, and TypeC campaigns run the longest, for 74.5 days.

2) Which was the most popular type of campaign?
![Untitled-1](https://github.com/galahad38/dunnhumby-coupon-redemption/assets/19240929/0d55d1f7-88cb-4299-bbd0-7398ad3d4c88)
TypeA was the most popular type of Campaign

3) Who were the most frequent shoppers?
![Untitled](https://github.com/galahad38/dunnhumby-coupon-redemption/assets/19240929/ebae2ecc-8338-4566-8621-0f9691d4b604)
The Data spans across a period of 2 years. This means that there were definitely customers/households that visited more than once a day, on average.

### Data Visualization:
Used Seaborn to visualize the distribution of individual variables (Univariate Analysis), as well as the relationship of the variables with the Target Variable (Bivariate Analysis)

Univariate Analysis:

Numerical Variables:
![Untitled](https://github.com/galahad38/dunnhumby-coupon-redemption/assets/19240929/95aca2ba-e48e-4a49-ae2c-469c59665b33)

Categorical Variables:
![Untitled](https://github.com/galahad38/dunnhumby-coupon-redemption/assets/19240929/9039cf73-2ed9-44b6-bc72-373b526d61d1)


Bivariate Analysis:

Numerical Variables vs Target:
![Untitled-1](https://github.com/galahad38/dunnhumby-coupon-redemption/assets/19240929/7aa5fd3a-88a5-4ade-9a2a-55efb81d44b7)


Categorical Variables vs Target:
![Untitled](https://github.com/galahad38/dunnhumby-coupon-redemption/assets/19240929/563ceeec-6e7d-4e9c-946a-73fa21301537)

### Treating Quasi-Separation:

1) Upon further inspection, 'distinct_coupons_redeemed_household' and 'coupon_success_ratio' both almost perfectly separate the subgroups in the target variable 'Redeemed'.
2) Logically, if the value of these columns is zero, then the target is 0, and if the value is non-zero, then the target is 1.
3) Since it doesn't make any sense to include these columns in the model, let us remove them.

### Further Binning the Categorical Variables:

![Untitled](https://github.com/galahad38/dunnhumby-coupon-redemption/assets/19240929/226678cf-9524-404a-b7ba-299370a78d11)

### Treating Multicollinearity:

Variables with the highest VIF (Variance Inflation Factor) were dropped from the DataFrame iteratively, until each variable in the DataFrame
had a VIF value lower than the chosen threshold of 10 (The maximum variation in the variable that can be explained using the other variables is 90%)

### Final Model:

Several classification models were built, including:
1) Logistic Regression
2) K-Nearest Neighbors
3) Bagged K-Nearest Neighbors
4) Decision Tree Classifier
5) Bagged Decision Tree Classifier
6) Random Forest Classifier
7) AdaBoost Classifier

The hyperparameters of the Random Forest Classifier were tuned, viz. 'criterion', 'n_estimators', 'max_depth', 'min_impurity_decrease', 'min_samples_split'.
The final model gave us a Macro Average F1-Score of 0.696.

### Future Scope:
I am satisfied with the outcome of this project. However it is simplistic and undoubtedly a prototype with huge scope for improvement:
1) Building a model on the Top 10 Most Important Features.
2) Performing Further Feature Engineering to obtain more significant Features.
3) Performing Oversampling Techniques to obtain more rows.
4) Product-Based Coupon Redemption instead of Customer-Based Coupon Redemption.

** Since Academic Requirements have been met for this project, I will put off implementing this for some time. In the meantime, feel free to contribute to this Project! **
