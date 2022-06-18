---
classes: wide

title: "Mode Validation"

excerpt: "You will test how good your model is."

categories:
- kaggle

tags:
- test
- split
- fit

comments: true

---


**This notebook is an exercise in the [Introduction to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) course.  You can reference the tutorial at [this link](https://www.kaggle.com/dansbecker/model-validation).**

---


## Recap
You've built a model. In this exercise you will test how good your model is.

Run the cell below to set up your coding environment where the previous exercise left off.


```python
# Code you have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex4 import *
print("Setup Complete")
```

    First in-sample predictions: [208500. 181500. 223500. 140000. 250000.]
    Actual target values for those homes: [208500, 181500, 223500, 140000, 250000]
    Setup Complete


# Exercises

## Step 1: Split Your Data
Use the `train_test_split` function to split up your data.

Give it the argument `random_state=1` so the `check` functions know what to expect when verifying your code.

Recall, your features are loaded in the DataFrame **X** and your target is loaded in **y**.



```python
# Import the train_test_split function and uncomment
from sklearn.model_selection import train_test_split

# fill in and uncomment
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Check your answer
step_1.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# The lines below will show you a hint or the solution.
# step_1.hint() 
# step_1.solution()

```

## Step 2: Specify and Fit the Model

Create a `DecisionTreeRegressor` model and fit it to the relevant data.
Set `random_state` to 1 again when creating the model.


```python
# You imported DecisionTreeRegressor in your last exercise
# and that code has been copied to the setup code above. So, no need to
# import it again

# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)

# Check your answer
step_2.check()
```

    [186500. 184000. 130000.  92000. 164500. 220000. 335000. 144152. 215000.
     262000.]
    [186500. 184000. 130000.  92000. 164500. 220000. 335000. 144152. 215000.
     262000.]



    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
step_2.hint()
step_2.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#3366cc">Hint:</span> Remember, you fit with training data. You will test with validation data soon



    <IPython.core.display.Javascript object>



<span style="color:#33cc99">Solution:</span> 
```python
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)
```


## Step 3: Make Predictions with Validation data



```python
# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

# Check your answer
step_3.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# step_3.hint()
# step_3.solution()
```

Inspect your predictions and actual values from validation data.


```python
# print the top few validation predictions
print(val_predictions[:5])
# print the top few actual prices from validation data
print(val_y[:5])
```

    [186500. 184000. 130000.  92000. 164500.]
    258     231500
    267     179500
    288     122000
    649      84500
    1233    142000
    Name: SalePrice, dtype: int64


What do you notice that is different from what you saw with in-sample predictions (which are printed after the top code cell in this page).

Do you remember why validation predictions differ from in-sample (or training) predictions? This is an important idea from the last lesson.

## Step 4: Calculate the Mean Absolute Error in Validation Data



```python
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)

# uncomment following line to see the validation_mae
print(val_mae)

# Check your answer
step_4.check()
```

    29652.931506849316



    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# step_4.hint()
# step_4.solution()
```

Is that MAE good?  There isn't a general rule for what values are good that applies across applications. But you'll see how to use (and improve) this number in the next step.

# Keep Going

You are ready for **[Underfitting and Overfitting](https://www.kaggle.com/dansbecker/underfitting-and-overfitting).**


---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/intro-to-machine-learning/discussion) to chat with other learners.*

{% if page.comments != false %}
{% include disqus.html %}
{% endif %}
