---
classes: wide

title: "Pandas grouping and Sorting"

excerpt: "In these exercises we'll apply groupwise analysis to our dataset."

categories:
- kaggle

tags:
- python
- pandas
- data

comments: true

---

**This notebook is an exercise in the [Pandas](https://www.kaggle.com/learn/pandas) course.  You can reference the tutorial at [this link](https://www.kaggle.com/residentmario/grouping-and-sorting).**

---


# Introduction

In these exercises we'll apply groupwise analysis to our dataset.

Run the code cell below to load the data before running the exercises.


```python
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
#pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.grouping_and_sorting import *
print("Setup complete.")
```

    Setup complete.


# Exercises

## 1.
Who are the most common wine reviewers in the dataset? Create a `Series` whose index is the `taster_twitter_handle` category from the dataset, and whose values count how many reviews each person wrote.


```python
reviews.columns
```




    Index(['country', 'description', 'designation', 'points', 'price', 'province',
           'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'title',
           'variety', 'winery'],
          dtype='object')




```python
reviews.taster_twitter_handle.head()
```




    0    @kerinokeefe
    1      @vossroger
    2     @paulgwine 
    3             NaN
    4     @paulgwine 
    Name: taster_twitter_handle, dtype: object




```python
reviews.groupby('taster_twitter_handle').describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">points</th>
      <th colspan="8" halign="left">price</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>taster_twitter_handle</th>
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
      <th>@AnneInVino</th>
      <td>3685.0</td>
      <td>90.562551</td>
      <td>2.373100</td>
      <td>80.0</td>
      <td>89.0</td>
      <td>90.0</td>
      <td>92.0</td>
      <td>97.0</td>
      <td>3398.0</td>
      <td>31.230135</td>
      <td>25.295871</td>
      <td>10.0</td>
      <td>19.0</td>
      <td>25.0</td>
      <td>38.00</td>
      <td>1100.0</td>
    </tr>
    <tr>
      <th>@JoeCz</th>
      <td>5147.0</td>
      <td>88.536235</td>
      <td>2.858701</td>
      <td>80.0</td>
      <td>87.0</td>
      <td>89.0</td>
      <td>91.0</td>
      <td>100.0</td>
      <td>5012.0</td>
      <td>35.175579</td>
      <td>44.434444</td>
      <td>5.0</td>
      <td>16.0</td>
      <td>22.0</td>
      <td>40.00</td>
      <td>850.0</td>
    </tr>
    <tr>
      <th>@bkfiona</th>
      <td>27.0</td>
      <td>86.888889</td>
      <td>1.739437</td>
      <td>82.0</td>
      <td>86.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>91.0</td>
      <td>27.0</td>
      <td>31.148148</td>
      <td>16.154789</td>
      <td>17.0</td>
      <td>22.5</td>
      <td>27.0</td>
      <td>35.00</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>@gordone_cellars</th>
      <td>4177.0</td>
      <td>88.626287</td>
      <td>2.698341</td>
      <td>80.0</td>
      <td>87.0</td>
      <td>89.0</td>
      <td>91.0</td>
      <td>97.0</td>
      <td>4171.0</td>
      <td>26.935507</td>
      <td>17.475901</td>
      <td>4.0</td>
      <td>15.0</td>
      <td>23.0</td>
      <td>32.00</td>
      <td>220.0</td>
    </tr>
    <tr>
      <th>@kerinokeefe</th>
      <td>10776.0</td>
      <td>88.867947</td>
      <td>2.474240</td>
      <td>80.0</td>
      <td>87.0</td>
      <td>89.0</td>
      <td>90.0</td>
      <td>100.0</td>
      <td>9874.0</td>
      <td>41.953413</td>
      <td>38.727135</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>50.00</td>
      <td>800.0</td>
    </tr>
    <tr>
      <th>@laurbuzz</th>
      <td>1835.0</td>
      <td>87.739510</td>
      <td>2.530672</td>
      <td>81.0</td>
      <td>86.0</td>
      <td>88.0</td>
      <td>90.0</td>
      <td>95.0</td>
      <td>1713.0</td>
      <td>24.492703</td>
      <td>21.967640</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>18.0</td>
      <td>28.00</td>
      <td>350.0</td>
    </tr>
    <tr>
      <th>@mattkettmann</th>
      <td>6332.0</td>
      <td>90.008686</td>
      <td>2.571257</td>
      <td>81.0</td>
      <td>88.0</td>
      <td>90.0</td>
      <td>92.0</td>
      <td>97.0</td>
      <td>6237.0</td>
      <td>38.642136</td>
      <td>31.994921</td>
      <td>7.0</td>
      <td>25.0</td>
      <td>35.0</td>
      <td>48.00</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>@paulgwine</th>
      <td>9532.0</td>
      <td>89.082564</td>
      <td>2.814445</td>
      <td>80.0</td>
      <td>87.0</td>
      <td>89.0</td>
      <td>91.0</td>
      <td>100.0</td>
      <td>9498.0</td>
      <td>33.644873</td>
      <td>18.936484</td>
      <td>5.0</td>
      <td>20.0</td>
      <td>29.0</td>
      <td>42.00</td>
      <td>275.0</td>
    </tr>
    <tr>
      <th>@suskostrzewa</th>
      <td>1085.0</td>
      <td>86.609217</td>
      <td>2.376140</td>
      <td>80.0</td>
      <td>85.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>94.0</td>
      <td>1073.0</td>
      <td>22.908667</td>
      <td>17.311163</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>19.0</td>
      <td>26.00</td>
      <td>320.0</td>
    </tr>
    <tr>
      <th>@vboone</th>
      <td>9537.0</td>
      <td>89.213379</td>
      <td>2.996796</td>
      <td>80.0</td>
      <td>87.0</td>
      <td>90.0</td>
      <td>91.0</td>
      <td>99.0</td>
      <td>9507.0</td>
      <td>46.621963</td>
      <td>32.655537</td>
      <td>7.0</td>
      <td>25.0</td>
      <td>39.0</td>
      <td>56.00</td>
      <td>625.0</td>
    </tr>
    <tr>
      <th>@vossroger</th>
      <td>25514.0</td>
      <td>88.708003</td>
      <td>3.036373</td>
      <td>80.0</td>
      <td>86.0</td>
      <td>88.0</td>
      <td>91.0</td>
      <td>100.0</td>
      <td>20172.0</td>
      <td>38.649960</td>
      <td>71.540473</td>
      <td>5.0</td>
      <td>15.0</td>
      <td>22.0</td>
      <td>40.00</td>
      <td>3300.0</td>
    </tr>
    <tr>
      <th>@wawinereport</th>
      <td>4966.0</td>
      <td>88.755739</td>
      <td>2.458547</td>
      <td>80.0</td>
      <td>87.0</td>
      <td>89.0</td>
      <td>91.0</td>
      <td>97.0</td>
      <td>4925.0</td>
      <td>34.085888</td>
      <td>20.029977</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>30.0</td>
      <td>42.00</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>@wineschach</th>
      <td>15134.0</td>
      <td>86.907493</td>
      <td>3.022859</td>
      <td>80.0</td>
      <td>85.0</td>
      <td>87.0</td>
      <td>89.0</td>
      <td>98.0</td>
      <td>14951.0</td>
      <td>25.231155</td>
      <td>28.723655</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>17.0</td>
      <td>25.00</td>
      <td>770.0</td>
    </tr>
    <tr>
      <th>@winewchristina</th>
      <td>6.0</td>
      <td>87.833333</td>
      <td>3.600926</td>
      <td>82.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>89.0</td>
      <td>93.0</td>
      <td>6.0</td>
      <td>29.333333</td>
      <td>11.165423</td>
      <td>19.0</td>
      <td>22.0</td>
      <td>28.5</td>
      <td>29.75</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>@worldwineguys</th>
      <td>1005.0</td>
      <td>88.719403</td>
      <td>2.044055</td>
      <td>82.0</td>
      <td>88.0</td>
      <td>89.0</td>
      <td>90.0</td>
      <td>97.0</td>
      <td>995.0</td>
      <td>25.238191</td>
      <td>19.984364</td>
      <td>6.0</td>
      <td>15.0</td>
      <td>20.0</td>
      <td>30.00</td>
      <td>320.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Your code here
reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()

# Check your answer
q1.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct:</span> 


```python
reviews_written = reviews.groupby('taster_twitter_handle').size()
```
or
```python
reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
```




```python
#q1.hint()
#q1.solution()
```

## 2.
What is the best wine I can buy for a given amount of money? Create a `Series` whose index is wine prices and whose values is the maximum number of points a wine costing that much was given in a review. Sort the values by price, ascending (so that `4.0` dollars is at the top and `3300.0` dollars is at the bottom).


```python
reviews.groupby('price').points.max()
```




    price
    4.0       86
    5.0       87
    6.0       88
    7.0       91
    8.0       91
              ..
    1900.0    98
    2000.0    97
    2013.0    91
    2500.0    96
    3300.0    88
    Name: points, Length: 390, dtype: int64




```python
best_rating_per_price = reviews.groupby('price').points.max()

# Check your answer
q2.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
#q2.hint()
#q2.solution()
```

## 3.
What are the minimum and maximum prices for each `variety` of wine? Create a `DataFrame` whose index is the `variety` category from the dataset and whose values are the `min` and `max` values thereof.


```python
reviews.groupby('variety').price.agg([min, max])
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
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>variety</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Abouriou</th>
      <td>15.0</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>Agiorgitiko</th>
      <td>10.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>Aglianico</th>
      <td>6.0</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>Aidani</th>
      <td>27.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>Airen</th>
      <td>8.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Zinfandel</th>
      <td>5.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Zlahtina</th>
      <td>13.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>Zweigelt</th>
      <td>9.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>Çalkarası</th>
      <td>19.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>Žilavka</th>
      <td>15.0</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
<p>707 rows × 2 columns</p>
</div>




```python
price_extremes = reviews.groupby('variety').price.agg([min, max])

# Check your answer
q3.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
#q3.hint()
#q3.solution()
```

## 4.
What are the most expensive wine varieties? Create a variable `sorted_varieties` containing a copy of the dataframe from the previous question where varieties are sorted in descending order based on minimum price, then on maximum price (to break ties).


```python
price_extremes.head()
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
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>variety</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Abouriou</th>
      <td>15.0</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>Agiorgitiko</th>
      <td>10.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>Aglianico</th>
      <td>6.0</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>Aidani</th>
      <td>27.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>Airen</th>
      <td>8.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sorted_varieties = price_extremes.sort_values(by=['min', 'max'],ascending=False)

# Check your answer
q4.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# q4.hint()
# q4.solution()
```

## 5.
Create a `Series` whose index is reviewers and whose values is the average review score given out by that reviewer. Hint: you will need the `taster_name` and `points` columns.


```python
reviews.groupby('taster_name').points.mean().sort_values()

```




    taster_name
    Alexander Peartree    85.855422
    Carrie Dykes          86.395683
    Susan Kostrzewa       86.609217
    Fiona Adams           86.888889
    Michael Schachner     86.907493
    Lauren Buzzeo         87.739510
    Christina Pickard     87.833333
    Jeff Jenssen          88.319756
    Anna Lee C. Iijima    88.415629
    Joe Czerwinski        88.536235
    Jim Gordon            88.626287
    Roger Voss            88.708003
    Sean P. Sullivan      88.755739
    Kerin O’Keefe         88.867947
    Paul Gregutt          89.082564
    Mike DeSimone         89.101167
    Virginie Boone        89.213379
    Matt Kettmann         90.008686
    Anne Krebiehl MW      90.562551
    Name: points, dtype: float64




```python
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()

# Check your answer
q5.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# q5.hint()
#q5.solution()
```

Are there significant differences in the average scores assigned by the various reviewers? Run the cell below to use the `describe()` method to see a summary of the range of values.


```python
reviewer_mean_ratings.describe()
```




    count    19.000000
    mean     88.233026
    std       1.243610
    min      85.855422
    25%      87.323501
    50%      88.536235
    75%      88.975256
    max      90.562551
    Name: points, dtype: float64



## 6.
What combination of countries and varieties are most common? Create a `Series` whose index is a `MultiIndex`of `{country, variety}` pairs. For example, a pinot noir produced in the US should map to `{"US", "Pinot Noir"}`. Sort the values in the `Series` in descending order based on wine count.


```python
reviews.groupby(['country', 'variety']).describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="8" halign="left">points</th>
      <th colspan="8" halign="left">price</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>country</th>
      <th>variety</th>
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
      <th rowspan="5" valign="top">Argentina</th>
      <th>Barbera</th>
      <td>1.0</td>
      <td>85.000000</td>
      <td>NaN</td>
      <td>85.0</td>
      <td>85.00</td>
      <td>85.0</td>
      <td>85.00</td>
      <td>85.0</td>
      <td>1.0</td>
      <td>18.000000</td>
      <td>NaN</td>
      <td>18.0</td>
      <td>18.00</td>
      <td>18.0</td>
      <td>18.00</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>Bonarda</th>
      <td>105.0</td>
      <td>86.504762</td>
      <td>2.587410</td>
      <td>80.0</td>
      <td>85.00</td>
      <td>87.0</td>
      <td>89.00</td>
      <td>92.0</td>
      <td>105.0</td>
      <td>16.628571</td>
      <td>6.367370</td>
      <td>9.0</td>
      <td>13.00</td>
      <td>15.0</td>
      <td>18.00</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>Bordeaux-style Red Blend</th>
      <td>89.0</td>
      <td>89.820225</td>
      <td>2.990750</td>
      <td>81.0</td>
      <td>88.00</td>
      <td>91.0</td>
      <td>92.00</td>
      <td>96.0</td>
      <td>86.0</td>
      <td>41.546512</td>
      <td>24.686580</td>
      <td>10.0</td>
      <td>25.00</td>
      <td>35.5</td>
      <td>50.00</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>Bordeaux-style White Blend</th>
      <td>1.0</td>
      <td>83.000000</td>
      <td>NaN</td>
      <td>83.0</td>
      <td>83.00</td>
      <td>83.0</td>
      <td>83.00</td>
      <td>83.0</td>
      <td>1.0</td>
      <td>14.000000</td>
      <td>NaN</td>
      <td>14.0</td>
      <td>14.00</td>
      <td>14.0</td>
      <td>14.00</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>Cabernet Blend</th>
      <td>8.0</td>
      <td>88.250000</td>
      <td>2.549510</td>
      <td>85.0</td>
      <td>86.75</td>
      <td>88.0</td>
      <td>89.25</td>
      <td>93.0</td>
      <td>8.0</td>
      <td>36.250000</td>
      <td>24.852422</td>
      <td>17.0</td>
      <td>24.75</td>
      <td>30.0</td>
      <td>35.75</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Uruguay</th>
      <th>Tannat-Cabernet Franc</th>
      <td>2.0</td>
      <td>90.000000</td>
      <td>1.414214</td>
      <td>89.0</td>
      <td>89.50</td>
      <td>90.0</td>
      <td>90.50</td>
      <td>91.0</td>
      <td>2.0</td>
      <td>19.000000</td>
      <td>4.242641</td>
      <td>16.0</td>
      <td>17.50</td>
      <td>19.0</td>
      <td>20.50</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>Tannat-Merlot</th>
      <td>6.0</td>
      <td>86.500000</td>
      <td>1.048809</td>
      <td>85.0</td>
      <td>86.00</td>
      <td>86.5</td>
      <td>87.00</td>
      <td>88.0</td>
      <td>6.0</td>
      <td>17.666667</td>
      <td>6.860515</td>
      <td>12.0</td>
      <td>13.00</td>
      <td>14.5</td>
      <td>21.25</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>Tannat-Syrah</th>
      <td>1.0</td>
      <td>84.000000</td>
      <td>NaN</td>
      <td>84.0</td>
      <td>84.00</td>
      <td>84.0</td>
      <td>84.00</td>
      <td>84.0</td>
      <td>1.0</td>
      <td>16.000000</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>16.00</td>
      <td>16.0</td>
      <td>16.00</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>Albariño</th>
      <td>6.0</td>
      <td>87.333333</td>
      <td>2.581989</td>
      <td>85.0</td>
      <td>85.25</td>
      <td>86.5</td>
      <td>89.25</td>
      <td>91.0</td>
      <td>6.0</td>
      <td>22.333333</td>
      <td>4.131182</td>
      <td>17.0</td>
      <td>19.00</td>
      <td>25.0</td>
      <td>25.00</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>Tempranillo-Tannat</th>
      <td>1.0</td>
      <td>88.000000</td>
      <td>NaN</td>
      <td>88.0</td>
      <td>88.00</td>
      <td>88.0</td>
      <td>88.00</td>
      <td>88.0</td>
      <td>1.0</td>
      <td>20.000000</td>
      <td>NaN</td>
      <td>20.0</td>
      <td>20.00</td>
      <td>20.0</td>
      <td>20.00</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
<p>1612 rows × 16 columns</p>
</div>




```python
reviews.groupby(['country', 'variety']).variety.count().sort_values(ascending=False)
```




    country  variety                 
    US       Pinot Noir                  9885
             Cabernet Sauvignon          7315
             Chardonnay                  6801
    France   Bordeaux-style Red Blend    4725
    Italy    Red Blend                   3624
                                         ... 
    Mexico   Cinsault                       1
             Grenache                       1
             Merlot                         1
             Rosado                         1
    Uruguay  White Blend                    1
    Name: variety, Length: 1612, dtype: int64




```python
country_variety_counts = reviews.groupby(['country', 'variety']).variety.count().sort_values(ascending=False)

# Check your answer
q6.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
#q6.hint()
#q6.solution()
```

# Keep going

Move on to the [**data types and missing data**](https://www.kaggle.com/residentmario/data-types-and-missing-values).

---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/pandas/discussion) to chat with other learners.*

{% if page.comments != false %}
{% include disqus.html %}
{% endif %}
