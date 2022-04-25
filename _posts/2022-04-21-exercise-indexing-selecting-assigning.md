---
classes: wide

title: "Pandas Indexing, Selecting and assigning"

excerpt: "In this set of exercises we will work with the Wine Reviews dataset."

categories:
- kaggle

tags:
- python
- pandas
- data

---

**This notebook is an exercise in the [Pandas](https://www.kaggle.com/learn/pandas) course.  You can reference the tutorial at [this link](https://www.kaggle.com/residentmario/indexing-selecting-assigning).**

---


# Introduction

In this set of exercises we will work with the [Wine Reviews dataset](https://www.kaggle.com/zynicide/wine-reviews). 

Run the following cell to load your data and some utility functions (including code to check your answers).


```python
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.indexing_selecting_and_assigning import *
print("Setup complete.")
```

    Setup complete.


Look at an overview of your data by running the following line.


```python
reviews.head()
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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
    </tr>
  </tbody>
</table>
</div>




```python
reviews.columns
```




    Index(['country', 'description', 'designation', 'points', 'price', 'province',
           'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'title',
           'variety', 'winery'],
          dtype='object')




```python
reviews.loc[reviews.country == 'Italy'].count()
```




    country        19540
    description    19540
                   ...  
    variety        19540
    winery         19540
    Length: 13, dtype: int64



# Exercises

## 1.

Select the `description` column from `reviews` and assign the result to the variable `desc`.


```python
# Your code here
desc = reviews['description']

# Check your answer
q1.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>


Follow-up question: what type of object is `desc`? If you're not sure, you can check by calling Python's `type` function: `type(desc)`.


```python
type(desc)
```




    pandas.core.series.Series




```python
#q1.hint()
#q1.solution()
```

## 2.

Select the first value from the description column of `reviews`, assigning it to variable `first_description`.


```python
first_description = desc[0]

# Check your answer
q2.check()
first_description
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct:</span> 


```python
first_description = reviews.description.iloc[0]
```
Note that while this is the preferred way to obtain the entry in the DataFrame, many other options will return a valid result, such as `reviews.description.loc[0]`, `reviews.description[0]`, and more!  






    "Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity."




```python
#q2.hint()
#q2.solution()
```

## 3. 

Select the first row of data (the first record) from `reviews`, assigning it to the variable `first_row`.


```python
first_row = reviews.iloc[0]

# Check your answer
q3.check()
first_row
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>





    country                                                    Italy
    description    Aromas include tropical fruit, broom, brimston...
                                         ...                        
    variety                                              White Blend
    winery                                                   Nicosia
    Name: 0, Length: 13, dtype: object




```python
#q3.hint()
#q3.solution()
```

## 4.

Select the first 10 values from the `description` column in `reviews`, assigning the result to variable `first_descriptions`.

Hint: format your output as a pandas Series.


```python
first_descriptions = reviews['description'].iloc[0:10]

# Check your answer
q4.check()
first_descriptions
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct:</span> 


```python
first_descriptions = reviews.description.iloc[:10]
```
Note that many other options will return a valid result, such as `desc.head(10)` and `reviews.loc[:9, "description"]`.    






    0    Aromas include tropical fruit, broom, brimston...
    1    This is ripe and fruity, a wine that is smooth...
                               ...                        
    8    Savory dried thyme notes accent sunnier flavor...
    9    This has great depth of flavor with its fresh ...
    Name: description, Length: 10, dtype: object




```python
#q4.hint()
#q4.solution()
```

## 5.

Select the records with index labels `1`, `2`, `3`, `5`, and `8`, assigning the result to the variable `sample_reviews`.

In other words, generate the following DataFrame:

![](https://i.imgur.com/sHZvI1O.png)


```python
sample_reviews = reviews.iloc[[1, 2, 3, 5, 8]]

# Check your answer
q5.check()
sample_reviews
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>





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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Spain</td>
      <td>Blackberry and raspberry aromas show a typical...</td>
      <td>Ars In Vitro</td>
      <td>87</td>
      <td>15.0</td>
      <td>Northern Spain</td>
      <td>Navarra</td>
      <td>NaN</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Tandem 2011 Ars In Vitro Tempranillo-Merlot (N...</td>
      <td>Tempranillo-Merlot</td>
      <td>Tandem</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Germany</td>
      <td>Savory dried thyme notes accent sunnier flavor...</td>
      <td>Shine</td>
      <td>87</td>
      <td>12.0</td>
      <td>Rheinhessen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Anna Lee C. Iijima</td>
      <td>NaN</td>
      <td>Heinz Eifel 2013 Shine Gewürztraminer (Rheinhe...</td>
      <td>Gewürztraminer</td>
      <td>Heinz Eifel</td>
    </tr>
  </tbody>
</table>
</div>




```python
#q5.hint()
#q5.solution()
```

## 6.

Create a variable `df` containing the `country`, `province`, `region_1`, and `region_2` columns of the records with the index labels `0`, `1`, `10`, and `100`. In other words, generate the following DataFrame:

![](https://i.imgur.com/FUCGiKP.png)


```python
df = reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]

# Check your answer
q6.check()
df
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>





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
      <th>country</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>US</td>
      <td>California</td>
      <td>Napa Valley</td>
      <td>Napa</td>
    </tr>
    <tr>
      <th>100</th>
      <td>US</td>
      <td>New York</td>
      <td>Finger Lakes</td>
      <td>Finger Lakes</td>
    </tr>
  </tbody>
</table>
</div>




```python
#q6.hint()
#q6.solution()
```

## 7.

Create a variable `df` containing the `country` and `variety` columns of the first 100 records. 

Hint: you may use `loc` or `iloc`. When working on the answer this question and the several of the ones that follow, keep the following "gotcha" described in the tutorial:

> `iloc` uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. 
`loc`, meanwhile, indexes inclusively. 

> This is particularly confusing when the DataFrame index is a simple numerical list, e.g. `0,...,1000`. In this case `df.iloc[0:1000]` will return 1000 entries, while `df.loc[0:1000]` return 1001 of them! To get 1000 elements using `loc`, you will need to go one lower and ask for `df.iloc[0:999]`. 


```python
df = reviews.loc[:99, ['country', 'variety']]

# Check your answer
q7.check()
df
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct:</span> 


```python
cols = ['country', 'variety']
df = reviews.loc[:99, cols]
```
or 
```python
cols_idx = [0, 11]
df = reviews.iloc[:100, cols_idx]
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
      <th>country</th>
      <th>variety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Italy</td>
      <td>White Blend</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Portugal</td>
      <td>Portuguese Red</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Italy</td>
      <td>Sangiovese</td>
    </tr>
    <tr>
      <th>99</th>
      <td>US</td>
      <td>Bordeaux-style Red Blend</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>




```python
#q7.hint()
#q7.solution()
```

## 8.

Create a DataFrame `italian_wines` containing reviews of wines made in `Italy`. Hint: `reviews.country` equals what?


```python
italian_wines = reviews[reviews.country == 'Italy']

# Check your answer
q8.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
#q8.hint()
#q8.solution()
```

## 9.

Create a DataFrame `top_oceania_wines` containing all reviews with at least 95 points (out of 100) for wines from Australia or New Zealand.


```python
top_oceania_wines = reviews[(reviews.points >= 95) & ((reviews.country == 'Australia') | (reviews.country == 'New Zealand'))]

# Check your answer
q9.check()
top_oceania_wines
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>





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
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>345</th>
      <td>Australia</td>
      <td>This wine contains some material over 100 year...</td>
      <td>Rare</td>
      <td>100</td>
      <td>350.0</td>
      <td>Victoria</td>
      <td>Rutherglen</td>
      <td>NaN</td>
      <td>Joe Czerwinski</td>
      <td>@JoeCz</td>
      <td>Chambers Rosewood Vineyards NV Rare Muscat (Ru...</td>
      <td>Muscat</td>
      <td>Chambers Rosewood Vineyards</td>
    </tr>
    <tr>
      <th>346</th>
      <td>Australia</td>
      <td>This deep brown wine smells like a damp, mossy...</td>
      <td>Rare</td>
      <td>98</td>
      <td>350.0</td>
      <td>Victoria</td>
      <td>Rutherglen</td>
      <td>NaN</td>
      <td>Joe Czerwinski</td>
      <td>@JoeCz</td>
      <td>Chambers Rosewood Vineyards NV Rare Muscadelle...</td>
      <td>Muscadelle</td>
      <td>Chambers Rosewood Vineyards</td>
    </tr>
    <tr>
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
    </tr>
    <tr>
      <th>122507</th>
      <td>New Zealand</td>
      <td>This blend of Cabernet Sauvignon (62.5%), Merl...</td>
      <td>SQM Gimblett Gravels Cabernets/Merlot</td>
      <td>95</td>
      <td>79.0</td>
      <td>Hawke's Bay</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Joe Czerwinski</td>
      <td>@JoeCz</td>
      <td>Squawking Magpie 2014 SQM Gimblett Gravels Cab...</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Squawking Magpie</td>
    </tr>
    <tr>
      <th>122939</th>
      <td>Australia</td>
      <td>Full-bodied and plush yet vibrant and imbued w...</td>
      <td>The Factor</td>
      <td>98</td>
      <td>125.0</td>
      <td>South Australia</td>
      <td>Barossa Valley</td>
      <td>NaN</td>
      <td>Joe Czerwinski</td>
      <td>@JoeCz</td>
      <td>Torbreck 2013 The Factor Shiraz (Barossa Valley)</td>
      <td>Shiraz</td>
      <td>Torbreck</td>
    </tr>
  </tbody>
</table>
<p>49 rows × 13 columns</p>
</div>




```python
#q9.hint()
#q9.solution()
```

# Keep going

Move on to learn about **[summary functions and maps](https://www.kaggle.com/residentmario/summary-functions-and-maps)**.

---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/pandas/discussion) to chat with other learners.*
