---
classes: wide

title: "Creating Featues"

excerpt: "In this exercise you'll start developing the features"

categories:
- kaggle

tags:
- data
- feature
- transforms
- counts
- building features
- breaking-down features

comments: true

---

**This notebook is an exercise in the [Feature Engineering](https://www.kaggle.com/learn/feature-engineering) course.  You can reference the tutorial at [this link](https://www.kaggle.com/ryanholbrook/creating-features).**

---


# Introduction #

In this exercise you'll start developing the features you identified in Exercise 2 as having the most potential. As you work through this exercise, you might take a moment to look at the data documentation again and consider whether the features we're creating make sense from a real-world perspective, and whether there are any useful combinations that stand out to you.

Run this cell to set everything up!


```python
# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering_new.ex3 import *

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


# Prepare data
df = pd.read_csv("../input/fe-course-data/ames.csv")
X = df.copy()
y = X.pop("SalePrice")
```

-------------------------------------------------------------------------------

Let's start with a few mathematical combinations. We'll focus on features describing areas -- having the same units (square-feet) makes it easy to combine them in sensible ways. Since we're using XGBoost (a tree-based model), we'll focus on ratios and sums.

# 1) Create Mathematical Transforms

Create the following features:

- `LivLotRatio`: the ratio of `GrLivArea` to `LotArea`
- `Spaciousness`: the sum of `FirstFlrSF` and `SecondFlrSF` divided by `TotRmsAbvGrd`
- `TotalOutsideSF`: the sum of `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `Threeseasonporch`, and `ScreenPorch`


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YearSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One_Story_1946_and_Newer_All_Styles</td>
      <td>Residential_Low_Density</td>
      <td>141.0</td>
      <td>31770.0</td>
      <td>Pave</td>
      <td>No_Alley_Access</td>
      <td>Slightly_Irregular</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No_Pool</td>
      <td>No_Fence</td>
      <td>None</td>
      <td>0.0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>One_Story_1946_and_Newer_All_Styles</td>
      <td>Residential_High_Density</td>
      <td>80.0</td>
      <td>11622.0</td>
      <td>Pave</td>
      <td>No_Alley_Access</td>
      <td>Regular</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>No_Pool</td>
      <td>Minimum_Privacy</td>
      <td>None</td>
      <td>0.0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>One_Story_1946_and_Newer_All_Styles</td>
      <td>Residential_Low_Density</td>
      <td>81.0</td>
      <td>14267.0</td>
      <td>Pave</td>
      <td>No_Alley_Access</td>
      <td>Slightly_Irregular</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No_Pool</td>
      <td>No_Fence</td>
      <td>Gar2</td>
      <td>12500.0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>One_Story_1946_and_Newer_All_Styles</td>
      <td>Residential_Low_Density</td>
      <td>93.0</td>
      <td>11160.0</td>
      <td>Pave</td>
      <td>No_Alley_Access</td>
      <td>Regular</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No_Pool</td>
      <td>No_Fence</td>
      <td>None</td>
      <td>0.0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Two_Story_1946_and_Newer</td>
      <td>Residential_Low_Density</td>
      <td>74.0</td>
      <td>13830.0</td>
      <td>Pave</td>
      <td>No_Alley_Access</td>
      <td>Slightly_Irregular</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>No_Pool</td>
      <td>Minimum_Privacy</td>
      <td>None</td>
      <td>0.0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 78 columns</p>
</div>




```python
X.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2930 entries, 0 to 2929
    Data columns (total 78 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   MSSubClass        2930 non-null   object 
     1   MSZoning          2930 non-null   object 
     2   LotFrontage       2930 non-null   float64
     3   LotArea           2930 non-null   float64
     4   Street            2930 non-null   object 
     5   Alley             2930 non-null   object 
     6   LotShape          2930 non-null   object 
     7   LandContour       2930 non-null   object 
     8   Utilities         2930 non-null   object 
     9   LotConfig         2930 non-null   object 
     10  LandSlope         2930 non-null   object 
     11  Neighborhood      2930 non-null   object 
     12  Condition1        2930 non-null   object 
     13  Condition2        2930 non-null   object 
     14  BldgType          2930 non-null   object 
     15  HouseStyle        2930 non-null   object 
     16  OverallQual       2930 non-null   object 
     17  OverallCond       2930 non-null   object 
     18  YearBuilt         2930 non-null   int64  
     19  YearRemodAdd      2930 non-null   int64  
     20  RoofStyle         2930 non-null   object 
     21  RoofMatl          2930 non-null   object 
     22  Exterior1st       2930 non-null   object 
     23  Exterior2nd       2930 non-null   object 
     24  MasVnrType        2930 non-null   object 
     25  MasVnrArea        2930 non-null   float64
     26  ExterQual         2930 non-null   object 
     27  ExterCond         2930 non-null   object 
     28  Foundation        2930 non-null   object 
     29  BsmtQual          2930 non-null   object 
     30  BsmtCond          2930 non-null   object 
     31  BsmtExposure      2930 non-null   object 
     32  BsmtFinType1      2930 non-null   object 
     33  BsmtFinSF1        2930 non-null   float64
     34  BsmtFinType2      2930 non-null   object 
     35  BsmtFinSF2        2930 non-null   float64
     36  BsmtUnfSF         2930 non-null   float64
     37  TotalBsmtSF       2930 non-null   float64
     38  Heating           2930 non-null   object 
     39  HeatingQC         2930 non-null   object 
     40  CentralAir        2930 non-null   object 
     41  Electrical        2930 non-null   object 
     42  FirstFlrSF        2930 non-null   float64
     43  SecondFlrSF       2930 non-null   float64
     44  LowQualFinSF      2930 non-null   float64
     45  GrLivArea         2930 non-null   float64
     46  BsmtFullBath      2930 non-null   int64  
     47  BsmtHalfBath      2930 non-null   int64  
     48  FullBath          2930 non-null   int64  
     49  HalfBath          2930 non-null   int64  
     50  BedroomAbvGr      2930 non-null   int64  
     51  KitchenAbvGr      2930 non-null   int64  
     52  KitchenQual       2930 non-null   object 
     53  TotRmsAbvGrd      2930 non-null   int64  
     54  Functional        2930 non-null   object 
     55  Fireplaces        2930 non-null   int64  
     56  FireplaceQu       2930 non-null   object 
     57  GarageType        2930 non-null   object 
     58  GarageFinish      2930 non-null   object 
     59  GarageCars        2930 non-null   int64  
     60  GarageArea        2930 non-null   float64
     61  GarageQual        2930 non-null   object 
     62  GarageCond        2930 non-null   object 
     63  PavedDrive        2930 non-null   object 
     64  WoodDeckSF        2930 non-null   float64
     65  OpenPorchSF       2930 non-null   float64
     66  EnclosedPorch     2930 non-null   float64
     67  Threeseasonporch  2930 non-null   float64
     68  ScreenPorch       2930 non-null   float64
     69  PoolArea          2930 non-null   float64
     70  PoolQC            2930 non-null   object 
     71  Fence             2930 non-null   object 
     72  MiscFeature       2930 non-null   object 
     73  MiscVal           2930 non-null   float64
     74  MoSold            2930 non-null   int64  
     75  YearSold          2930 non-null   int64  
     76  SaleType          2930 non-null   object 
     77  SaleCondition     2930 non-null   object 
    dtypes: float64(19), int64(13), object(46)
    memory usage: 1.7+ MB



```python
# YOUR CODE HERE
X_1 = pd.DataFrame()  # dataframe to hold new features

sf_columns = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "Threeseasonporch", "ScreenPorch"]

X_1["LivLotRatio"] = X.GrLivArea / X.LotArea
X_1["Spaciousness"] = (X.FirstFlrSF + X.SecondFlrSF) / X.TotRmsAbvGrd
X_1["TotalOutsideSF"] = X[sf_columns].sum(axis=1)


# Check your answer
q_1.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# Lines below will give you a hint or solution code
# q_1.hint()
#q_1.solution()
```


```python
X_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LivLotRatio</th>
      <th>Spaciousness</th>
      <th>TotalOutsideSF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.052125</td>
      <td>236.571429</td>
      <td>272.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.077095</td>
      <td>179.200000</td>
      <td>260.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.093152</td>
      <td>221.500000</td>
      <td>429.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.189068</td>
      <td>263.750000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.117787</td>
      <td>271.500000</td>
      <td>246.0</td>
    </tr>
  </tbody>
</table>
</div>



-------------------------------------------------------------------------------

If you've discovered an interaction effect between a numeric feature and a categorical feature, you might want to model it explicitly using a one-hot encoding, like so:

```
# One-hot encode Categorical feature, adding a column prefix "Cat"
X_new = pd.get_dummies(df.Categorical, prefix="Cat")

# Multiply row-by-row
X_new = X_new.mul(df.Continuous, axis=0)

# Join the new features to the feature set
X = X.join(X_new)
```

# 2) Interaction with a Categorical

We discovered an interaction between `BldgType` and `GrLivArea` in Exercise 2. Now create their interaction features.


```python
# YOUR CODE HERE
# One-hot encode BldgType. Use `prefix="Bldg"` in `get_dummies`
X_2 = pd.get_dummies(X.BldgType, prefix="Bldg")
# Multiply
X_2 = X_2.mul(X.GrLivArea, axis=0)


# Check your answer
q_2.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
X_2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2930 entries, 0 to 2929
    Data columns (total 5 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Bldg_Duplex    2930 non-null   float64
     1   Bldg_OneFam    2930 non-null   float64
     2   Bldg_Twnhs     2930 non-null   float64
     3   Bldg_TwnhsE    2930 non-null   float64
     4   Bldg_TwoFmCon  2930 non-null   float64
    dtypes: float64(5)
    memory usage: 114.6 KB



```python
X_2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bldg_Duplex</th>
      <th>Bldg_OneFam</th>
      <th>Bldg_Twnhs</th>
      <th>Bldg_TwnhsE</th>
      <th>Bldg_TwoFmCon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1656.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>896.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1329.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>2110.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1629.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Lines below will give you a hint or solution code
q_2.hint()
#q_2.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#3366cc">Hint:</span> Your code should look something like:
```python
X_2 = pd.get_dummies(____, prefix="Bldg")
X_2 = X_2.mul(____, axis=0)
```



# 3) Count Feature

Let's try creating a feature that describes how many kinds of outdoor areas a dwelling has. Create a feature `PorchTypes` that counts how many of the following are greater than 0.0:

```
WoodDeckSF
OpenPorchSF
EnclosedPorch
Threeseasonporch
ScreenPorch
```


```python
X_3 = pd.DataFrame()

# YOUR CODE HERE
porch_columns = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "Threeseasonporch", "ScreenPorch"]
X_3["PorchTypes"] = X[porch_columns].gt(0).sum(axis=1)


# Check your answer
q_3.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# Lines below will give you a hint or solution code
#q_3.hint()
#q_3.solution()
```

# 4) Break Down a Categorical Feature

`MSSubClass` describes the type of a dwelling:


```python
df.MSSubClass.unique()
```




    array(['One_Story_1946_and_Newer_All_Styles', 'Two_Story_1946_and_Newer',
           'One_Story_PUD_1946_and_Newer',
           'One_and_Half_Story_Finished_All_Ages', 'Split_Foyer',
           'Two_Story_PUD_1946_and_Newer', 'Split_or_Multilevel',
           'One_Story_1945_and_Older', 'Duplex_All_Styles_and_Ages',
           'Two_Family_conversion_All_Styles_and_Ages',
           'One_and_Half_Story_Unfinished_All_Ages',
           'Two_Story_1945_and_Older', 'Two_and_Half_Story_All_Ages',
           'One_Story_with_Finished_Attic_All_Ages',
           'PUD_Multilevel_Split_Level_Foyer',
           'One_and_Half_Story_PUD_All_Ages'], dtype=object)



You can see that there is a more general categorization described (roughly) by the first word of each category. Create a feature containing only these first words by splitting `MSSubClass` at the first underscore `_`. (Hint: In the `split` method use an argument `n=1`.)


```python
X_4 = pd.DataFrame()

# YOUR CODE HERE
X_4["MSClass"] = df.MSSubClass.str.split(pat="_", n=1, expand=True)[0]

# Check your answer
q_4.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# Lines below will give you a hint or solution code
q_4.hint()
#q_4.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#3366cc">Hint:</span> Your code should look something like:
```python
X_4 = pd.DataFrame()

X_4["MSClass"] = df.____.str.____(____, n=1, expand=True)[____]
```



# 5) Use a Grouped Transform

The value of a home often depends on how it compares to typical homes in its neighborhood. Create a feature `MedNhbdArea` that describes the *median* of `GrLivArea` grouped on `Neighborhood`.


```python
X_5 = pd.DataFrame()

# YOUR CODE HERE
X_5["MedNhbdArea"] = (
    X.groupby("Neighborhood")
    ["GrLivArea"]
    .transform("median"))

# Check your answer
q_5.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
X_5.MedNhbdArea.unique()
```




    array([1200. , 1560. , 1767. , 1632. , 1555. , 1092. , 1322. , 1832. ,
           1455.5, 2418. , 1575. , 1052. , 1226. , 1231. , 1374. , 1128. ,
           1694. , 1536.5, 1195.5, 1504. , 1648. , 1118. , 1282. , 1650.5,
           1706.5, 1398.5, 1320. ])




```python
# Lines below will give you a hint or solution code
#q_5.hint()
#q_5.solution()
```

Now you've made your first new feature set! If you like, you can run the cell below to score the model with all of your new features added:


```python
X_new = X.join([X_1, X_2, X_3, X_4, X_5])
score_dataset(X_new, y)
```




    0.13847331710099203



# Keep Going #

[**Untangle spatial relationships**](https://www.kaggle.com/ryanholbrook/clustering-with-k-means) by adding cluster labels to your dataset.

---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/feature-engineering/discussion) to chat with other learners.*

{% if page.comments != false %}
{% include disqus.html %}
{% endif %}
