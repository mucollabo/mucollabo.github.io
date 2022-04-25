---
title: "Pandas Summary, Functions and Maps"

excerpt: "Now you are ready to get a deeper understanding of your data."

categories:
- kaggle

tags:
- python
- pandas
- data

---


**This notebook is an exercise in the [Pandas](https://www.kaggle.com/learn/pandas) course.  You can reference the tutorial at [this link](https://www.kaggle.com/residentmario/summary-functions-and-maps).**

---


# Introduction

Now you are ready to get a deeper understanding of your data.

Run the following cell to load your data and some utility functions (including code to check your answers).


```python
import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()
```

    Setup complete.





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



# Exercises

## 1.

What is the median of the `points` column in the `reviews` DataFrame?


```python
median_points = reviews.points.median()

# Check your answer
q1.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
#q1.hint()
#q1.solution()
```

## 2. 
What countries are represented in the dataset? (Your answer should not include any duplicates.)


```python
reviews.country.unique()
```




    array(['Italy', 'Portugal', 'US', 'Spain', 'France', 'Germany',
           'Argentina', 'Chile', 'Australia', 'Austria', 'South Africa',
           'New Zealand', 'Israel', 'Hungary', 'Greece', 'Romania', 'Mexico',
           'Canada', nan, 'Turkey', 'Czech Republic', 'Slovenia',
           'Luxembourg', 'Croatia', 'Georgia', 'Uruguay', 'England',
           'Lebanon', 'Serbia', 'Brazil', 'Moldova', 'Morocco', 'Peru',
           'India', 'Bulgaria', 'Cyprus', 'Armenia', 'Switzerland',
           'Bosnia and Herzegovina', 'Ukraine', 'Slovakia', 'Macedonia',
           'China', 'Egypt'], dtype=object)




```python
countries = reviews.country.unique()

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
How often does each country appear in the dataset? Create a Series `reviews_per_country` mapping countries to the count of reviews of wines from that country.


```python
reviews_per_country = reviews.country.value_counts()

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
Create variable `centered_price` containing a version of the `price` column with the mean price subtracted.

(Note: this 'centering' transformation is a common preprocessing step before applying various machine learning algorithms.) 


```python
centered_price = reviews.price - reviews.price.mean()

# Check your answer
q4.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
#q4.hint()
#q4.solution()
```

## 5.
I'm an economical wine buyer. Which wine is the "best bargain"? Create a variable `bargain_wine` with the title of the wine with the highest points-to-price ratio in the dataset.


```python
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']

# Check your answer
q5.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
bargain_wine
```




    'Bandit NV Merlot (California)'




```python
badwine_idx = (reviews.points / reviews.price).idxmin()
bad_wine = reviews.loc[badwine_idx, 'title']
```


```python
bad_wine
```




    'Château les Ormes Sorbet 2013  Médoc'




```python
q5.hint()
q5.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#3366cc">Hint:</span> The `idxmax` method may be useful here.



    <IPython.core.display.Javascript object>



<span style="color:#33cc99">Solution:</span> 
```python
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']
```


## 6.
There are only so many words you can use when describing a bottle of wine. Is a wine more likely to be "tropical" or "fruity"? Create a Series `descriptor_counts` counting how many times each of these two words appears in the `description` column in the dataset. (For simplicity, let's ignore the capitalized versions of these words.)


```python
reviews.columns
```




    Index(['country', 'description', 'designation', 'points', 'price', 'province',
           'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'title',
           'variety', 'winery'],
          dtype='object')




```python
reviews.description.count()
```




    129971




```python
n_trop = reviews.description.map(lambda desc: 'tropical' in desc).sum()
n_fruity = reviews.description.map(lambda desc: 'fruity' in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

# Check your answer
q6.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
q6.hint()
q6.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#3366cc">Hint:</span> Use a map to check each description for the string `tropical`, then count up the number of times this is `True`. Repeat this for `fruity`. Finally, create a `Series` combining the two values.



    <IPython.core.display.Javascript object>



<span style="color:#33cc99">Solution:</span> 
```python
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])
```


## 7.
We'd like to host these wine reviews on our website, but a rating system ranging from 80 to 100 points is too hard to understand - we'd like to translate them into simple star ratings. A score of 95 or higher counts as 3 stars, a score of at least 85 but less than 95 is 2 stars. Any other score is 1 star.

Also, the Canadian Vintners Association bought a lot of ads on the site, so any wines from Canada should automatically get 3 stars, regardless of points.

Create a series `star_ratings` with the number of stars corresponding to each review in the dataset.


```python
type(reviews)
```




    pandas.core.frame.DataFrame




```python
def make_star(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85 and row.points <95:
        return 2
    else:
        return 1
        

star_ratings = reviews.apply(make_star, axis='columns')

# Check your answer
q7.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
star_ratings.describe()
```




    count    129971.000000
    mean          1.924999
                 ...      
    75%           2.000000
    max           3.000000
    Length: 8, dtype: float64




```python
q7.hint()
q7.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#3366cc">Hint:</span> Begin by writing a custom function that accepts a row from the DataFrame as input and returns the star rating corresponding to the row.  Then, use `DataFrame.apply` to apply the custom function to every row in the dataset.



    <IPython.core.display.Javascript object>



<span style="color:#33cc99">Solution:</span> 
```python
def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1
    
star_ratings = reviews.apply(stars, axis='columns')
```


# Keep going
Continue to **[grouping and sorting](https://www.kaggle.com/residentmario/grouping-and-sorting)**.

---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/pandas/discussion) to chat with other learners.*
