---
classes: wide

title: "IntermediateML pipelines"

categories:
- kaggle

tags:
- data
- feature
- one-hot-encoding
- pipeline
- strategy

comments: true

---


**This notebook is an exercise in the [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) course.  You can reference the tutorial at [this link](https://www.kaggle.com/alexisbcook/pipelines).**

---


In this exercise, you will use **pipelines** to improve the efficiency of your machine learning code.

# Setup

The questions below will give you feedback on your work. Run the following cell to set up the feedback system.


```python
# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex4 import *
print("Setup Complete")
```

    Setup Complete


You will work with data from the [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course). 

![Ames Housing dataset image](https://i.imgur.com/lTJVG4e.png)

Run the next code cell without changes to load the training and validation sets in `X_train`, `X_valid`, `y_train`, and `y_valid`.  The test set is loaded in `X_test`.


```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
```


```python
X_train.head()
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
      <th>MSZoning</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>...</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>619</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>...</td>
      <td>774</td>
      <td>0</td>
      <td>108</td>
      <td>0</td>
      <td>0</td>
      <td>260</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>871</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>PosN</td>
      <td>Norm</td>
      <td>...</td>
      <td>308</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>93</th>
      <td>RL</td>
      <td>Pave</td>
      <td>Grvl</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>...</td>
      <td>432</td>
      <td>0</td>
      <td>0</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>818</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>CulDSac</td>
      <td>Gtl</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>...</td>
      <td>857</td>
      <td>150</td>
      <td>59</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>303</th>
      <td>RL</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>...</td>
      <td>843</td>
      <td>468</td>
      <td>81</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 76 columns</p>
</div>



The next code cell uses code from the tutorial to preprocess the data and train a model.  Run this code without changes.


```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))
```

    MAE: 17861.780102739725


The code yields a value around 17862 for the mean absolute error (MAE).  In the next step, you will amend the code to do better.

# Step 1: Improve the performance

### Part A

Now, it's your turn!  In the code cell below, define your own preprocessing steps and random forest model.  Fill in values for the following variables:
- `numerical_transformer`
- `categorical_transformer`
- `model`

To pass this part of the exercise, you need only define valid preprocessing steps and a random forest model.


```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1168 entries, 619 to 685
    Data columns (total 76 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   MSZoning       1168 non-null   object 
     1   Street         1168 non-null   object 
     2   Alley          71 non-null     object 
     3   LotShape       1168 non-null   object 
     4   LandContour    1168 non-null   object 
     5   Utilities      1168 non-null   object 
     6   LotConfig      1168 non-null   object 
     7   LandSlope      1168 non-null   object 
     8   Condition1     1168 non-null   object 
     9   Condition2     1168 non-null   object 
     10  BldgType       1168 non-null   object 
     11  HouseStyle     1168 non-null   object 
     12  RoofStyle      1168 non-null   object 
     13  RoofMatl       1168 non-null   object 
     14  MasVnrType     1162 non-null   object 
     15  ExterQual      1168 non-null   object 
     16  ExterCond      1168 non-null   object 
     17  Foundation     1168 non-null   object 
     18  BsmtQual       1140 non-null   object 
     19  BsmtCond       1140 non-null   object 
     20  BsmtExposure   1140 non-null   object 
     21  BsmtFinType1   1140 non-null   object 
     22  BsmtFinType2   1139 non-null   object 
     23  Heating        1168 non-null   object 
     24  HeatingQC      1168 non-null   object 
     25  CentralAir     1168 non-null   object 
     26  Electrical     1167 non-null   object 
     27  KitchenQual    1168 non-null   object 
     28  Functional     1168 non-null   object 
     29  FireplaceQu    617 non-null    object 
     30  GarageType     1110 non-null   object 
     31  GarageFinish   1110 non-null   object 
     32  GarageQual     1110 non-null   object 
     33  GarageCond     1110 non-null   object 
     34  PavedDrive     1168 non-null   object 
     35  PoolQC         4 non-null      object 
     36  Fence          214 non-null    object 
     37  MiscFeature    49 non-null     object 
     38  SaleType       1168 non-null   object 
     39  SaleCondition  1168 non-null   object 
     40  MSSubClass     1168 non-null   int64  
     41  LotFrontage    956 non-null    float64
     42  LotArea        1168 non-null   int64  
     43  OverallQual    1168 non-null   int64  
     44  OverallCond    1168 non-null   int64  
     45  YearBuilt      1168 non-null   int64  
     46  YearRemodAdd   1168 non-null   int64  
     47  MasVnrArea     1162 non-null   float64
     48  BsmtFinSF1     1168 non-null   int64  
     49  BsmtFinSF2     1168 non-null   int64  
     50  BsmtUnfSF      1168 non-null   int64  
     51  TotalBsmtSF    1168 non-null   int64  
     52  1stFlrSF       1168 non-null   int64  
     53  2ndFlrSF       1168 non-null   int64  
     54  LowQualFinSF   1168 non-null   int64  
     55  GrLivArea      1168 non-null   int64  
     56  BsmtFullBath   1168 non-null   int64  
     57  BsmtHalfBath   1168 non-null   int64  
     58  FullBath       1168 non-null   int64  
     59  HalfBath       1168 non-null   int64  
     60  BedroomAbvGr   1168 non-null   int64  
     61  KitchenAbvGr   1168 non-null   int64  
     62  TotRmsAbvGrd   1168 non-null   int64  
     63  Fireplaces     1168 non-null   int64  
     64  GarageYrBlt    1110 non-null   float64
     65  GarageCars     1168 non-null   int64  
     66  GarageArea     1168 non-null   int64  
     67  WoodDeckSF     1168 non-null   int64  
     68  OpenPorchSF    1168 non-null   int64  
     69  EnclosedPorch  1168 non-null   int64  
     70  3SsnPorch      1168 non-null   int64  
     71  ScreenPorch    1168 non-null   int64  
     72  PoolArea       1168 non-null   int64  
     73  MiscVal        1168 non-null   int64  
     74  MoSold         1168 non-null   int64  
     75  YrSold         1168 non-null   int64  
    dtypes: float64(3), int64(33), object(40)
    memory usage: 702.6+ KB



```python
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Check your answer
step_1.a.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# Lines below will give you a hint or solution code
step_1.a.hint()
#step_1.a.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#3366cc">Hint:</span> While there are many different potential solutions to this problem, we achieved satisfactory results by changing only `column_transformer` from the default value - specifically, we changed the `strategy` parameter that decides how missing values are imputed.


### Part B

Run the code cell below without changes.

To pass this step, you need to have defined a pipeline in **Part A** that achieves lower MAE than the code above.  You're encouraged to take your time here and try out many different approaches, to see how low you can get the MAE!  (_If your code does not pass, please amend the preprocessing steps and model in Part A._)


```python
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

# Check your answer
step_1.b.check()
```

    MAE: 17479.87044520548



    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# Line below will give you a hint
#step_1.b.hint()
```

# Step 2: Generate test predictions

Now, you'll use your trained model to generate predictions with the test data.


```python
# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)

# Check your answer
step_2.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# Lines below will give you a hint or solution code
#step_2.hint()
#step_2.solution()
```

Run the next code cell without changes to save your results to a CSV file that can be submitted directly to the competition.


```python
# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
```


```python
output.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 2 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Id         1459 non-null   int64  
     1   SalePrice  1459 non-null   float64
    dtypes: float64(1), int64(1)
    memory usage: 22.9 KB


# Submit your results

Once you have successfully completed Step 2, you're ready to submit your results to the leaderboard!  If you choose to do so, make sure that you have already joined the competition by clicking on the **Join Competition** button at [this link](https://www.kaggle.com/c/home-data-for-ml-course).  
1. Begin by clicking on the **Save Version** button in the top right corner of the window.  This will generate a pop-up window.  
2. Ensure that the **Save and Run All** option is selected, and then click on the **Save** button.
3. This generates a window in the bottom left corner of the notebook.  After it has finished running, click on the number to the right of the **Save Version** button.  This pulls up a list of versions on the right of the screen.  Click on the ellipsis **(...)** to the right of the most recent version, and select **Open in Viewer**.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
4. Click on the **Output** tab on the right of the screen.  Then, click on the file you would like to submit, and click on the **Submit** button to submit your results to the leaderboard.

You have now successfully submitted to the competition!

If you want to keep working to improve your performance, select the **Edit** button in the top right of the screen. Then you can change your code and repeat the process. There's a lot of room to improve, and you will climb up the leaderboard as you work.


# Keep going

Move on to learn about [**cross-validation**](https://www.kaggle.com/alexisbcook/cross-validation), a technique you can use to obtain more accurate estimates of model performance!

---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/intermediate-machine-learning/discussion) to chat with other learners.*

{% if page.comments != false %}
{% include disqus.html %}
{% endif %}
