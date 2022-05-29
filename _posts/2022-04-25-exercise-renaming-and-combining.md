---
classes: wide

title: "Pandas Renaming and Combining"

excerpt: "Run the following cell to load your data and some utility functions."

categories:
- kaggle

tags:
- python
- pandas
- data

comments: true

---

**This notebook is an exercise in the [Pandas](https://www.kaggle.com/learn/pandas) course.  You can reference the tutorial at [this link](https://www.kaggle.com/residentmario/renaming-and-combining).**

---


# Introduction

Run the following cell to load your data and some utility functions.


```python
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.renaming_and_combining import *
print("Setup complete.")
```

    Setup complete.


# Exercises

View the first several lines of your data by running the cell below:


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



## 1.
`region_1` and `region_2` are pretty uninformative names for locale columns in the dataset. Create a copy of `reviews` with these columns renamed to `region` and `locale`, respectively.


```python
# Your code here
renamed = reviews.rename(columns = {'region_1':'region', 'region_2':'locale'})

# Check your answer
q1.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
renamed.head()
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
      <th>region</th>
      <th>locale</th>
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
#q1.hint()
#q1.solution()
```

## 2.
Set the index name in the dataset to `wines`.


```python
reindexed = reviews.rename_axis("wines", axis='rows')

# Check your answer
q2.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
reindexed.head()
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
    <tr>
      <th>wines</th>
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
#q2.hint()
#q2.solution()
```

## 3.
The [Things on Reddit](https://www.kaggle.com/residentmario/things-on-reddit/data) dataset includes product links from a selection of top-ranked forums ("subreddits") on reddit.com. Run the cell below to load a dataframe of products mentioned on the */r/gaming* subreddit and another dataframe for products mentioned on the *r//movies* subreddit.


```python
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
```

Create a `DataFrame` of products mentioned on *either* subreddit.


```python
gaming_products.head()
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
      <th>name</th>
      <th>category</th>
      <th>amazon_link</th>
      <th>total_mentions</th>
      <th>subreddit_mentions</th>
      <th>subreddit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BOOMco Halo Covenant Needler Blaster</td>
      <td>Toys &amp; Games</td>
      <td>https://www.amazon.com/BOOMco-Halo-Covenant-Ne...</td>
      <td>4.0</td>
      <td>4</td>
      <td>r/gaming</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Raspberry PI 3 Model B 1.2GHz 64-bit quad-core...</td>
      <td>Electronics</td>
      <td>https://www.amazon.com/Raspberry-Model-A1-2GHz...</td>
      <td>19.0</td>
      <td>3</td>
      <td>r/gaming</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CanaKit 5V 2.5A Raspberry Pi 3 Power Supply / ...</td>
      <td>Electronics</td>
      <td>https://www.amazon.com/CanaKit-Raspberry-Suppl...</td>
      <td>7.0</td>
      <td>3</td>
      <td>r/gaming</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Panasonic K-KJ17MCA4BA Advanced Individual Cel...</td>
      <td>Electronics</td>
      <td>https://www.amazon.com/Panasonic-Advanced-Indi...</td>
      <td>29.0</td>
      <td>2</td>
      <td>r/gaming</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mayflash GameCube Controller Adapter for Wii U...</td>
      <td>Electronics</td>
      <td>https://www.amazon.com/GameCube-Controller-Ada...</td>
      <td>24.0</td>
      <td>2</td>
      <td>r/gaming</td>
    </tr>
  </tbody>
</table>
</div>




```python
gaming_products.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 493 entries, 0 to 492
    Data columns (total 6 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   name                493 non-null    object 
     1   category            493 non-null    object 
     2   amazon_link         493 non-null    object 
     3   total_mentions      491 non-null    float64
     4   subreddit_mentions  493 non-null    int64  
     5   subreddit           493 non-null    object 
    dtypes: float64(1), int64(1), object(4)
    memory usage: 23.2+ KB



```python
gaming_products.describe()
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
      <th>total_mentions</th>
      <th>subreddit_mentions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>491.000000</td>
      <td>493.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.853360</td>
      <td>1.064909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.504293</td>
      <td>0.284858</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>46.000000</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
movie_products.head()
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
      <th>name</th>
      <th>category</th>
      <th>amazon_link</th>
      <th>total_mentions</th>
      <th>subreddit_mentions</th>
      <th>subreddit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Marvel Cinematic Universe: Phase One - Avenger...</td>
      <td>Movies &amp; TV</td>
      <td>https://www.amazon.com/Marvel-Cinematic-Univer...</td>
      <td>4.0</td>
      <td>3</td>
      <td>r/movies</td>
    </tr>
    <tr>
      <th>1</th>
      <td>On Stranger Tides</td>
      <td>Books</td>
      <td>https://www.amazon.com/Stranger-Tides-Tim-Powe...</td>
      <td>3.0</td>
      <td>3</td>
      <td>r/movies</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Superintelligence: Paths, Dangers, Strategies</td>
      <td>Books</td>
      <td>https://www.amazon.com/Superintelligence-Dange...</td>
      <td>7.0</td>
      <td>2</td>
      <td>r/movies</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Secret History of Star Wars</td>
      <td>Books</td>
      <td>https://www.amazon.com/Secret-History-Star-War...</td>
      <td>4.0</td>
      <td>2</td>
      <td>r/movies</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2D Glasses 4 Pack - Turns 3D movies back into ...</td>
      <td>Electronics</td>
      <td>https://www.amazon.com/gp/product/B00K9E7GCC</td>
      <td>3.0</td>
      <td>2</td>
      <td>r/movies</td>
    </tr>
  </tbody>
</table>
</div>




```python
movie_products.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 303 entries, 0 to 302
    Data columns (total 6 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   name                303 non-null    object 
     1   category            303 non-null    object 
     2   amazon_link         303 non-null    object 
     3   total_mentions      302 non-null    float64
     4   subreddit_mentions  303 non-null    int64  
     5   subreddit           303 non-null    object 
    dtypes: float64(1), int64(1), object(4)
    memory usage: 14.3+ KB



```python
movie_products.describe()
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
      <th>total_mentions</th>
      <th>subreddit_mentions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>302.000000</td>
      <td>303.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.470199</td>
      <td>1.046205</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.131300</td>
      <td>0.239710</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
combined_products = pd.concat([gaming_products, movie_products])

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
The [Powerlifting Database](https://www.kaggle.com/open-powerlifting/powerlifting-database) dataset on Kaggle includes one CSV table for powerlifting meets and a separate one for powerlifting competitors. Run the cell below to load these datasets into dataframes:


```python
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
```


```python
powerlifting_meets.head()
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
      <th>MeetID</th>
      <th>MeetPath</th>
      <th>Federation</th>
      <th>Date</th>
      <th>MeetCountry</th>
      <th>MeetState</th>
      <th>MeetTown</th>
      <th>MeetName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>365strong/1601</td>
      <td>365Strong</td>
      <td>2016-10-29</td>
      <td>USA</td>
      <td>NC</td>
      <td>Charlotte</td>
      <td>2016 Junior &amp; Senior National Powerlifting Cha...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>365strong/1602</td>
      <td>365Strong</td>
      <td>2016-11-19</td>
      <td>USA</td>
      <td>MO</td>
      <td>Ozark</td>
      <td>Thanksgiving Powerlifting Classic</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>365strong/1603</td>
      <td>365Strong</td>
      <td>2016-07-09</td>
      <td>USA</td>
      <td>NC</td>
      <td>Charlotte</td>
      <td>Charlotte Europa Games</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>365strong/1604</td>
      <td>365Strong</td>
      <td>2016-06-11</td>
      <td>USA</td>
      <td>SC</td>
      <td>Rock Hill</td>
      <td>Carolina Cup Push Pull Challenge</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>365strong/1605</td>
      <td>365Strong</td>
      <td>2016-04-10</td>
      <td>USA</td>
      <td>SC</td>
      <td>Rock Hill</td>
      <td>Eastern USA Challenge</td>
    </tr>
  </tbody>
</table>
</div>




```python
powerlifting_meets.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8482 entries, 0 to 8481
    Data columns (total 8 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   MeetID       8482 non-null   int64 
     1   MeetPath     8482 non-null   object
     2   Federation   8482 non-null   object
     3   Date         8482 non-null   object
     4   MeetCountry  8482 non-null   object
     5   MeetState    5496 non-null   object
     6   MeetTown     6973 non-null   object
     7   MeetName     8482 non-null   object
    dtypes: int64(1), object(7)
    memory usage: 530.2+ KB



```python
powerlifting_competitors.head()
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
      <th>MeetID</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Equipment</th>
      <th>Age</th>
      <th>Division</th>
      <th>BodyweightKg</th>
      <th>WeightClassKg</th>
      <th>Squat4Kg</th>
      <th>BestSquatKg</th>
      <th>Bench4Kg</th>
      <th>BestBenchKg</th>
      <th>Deadlift4Kg</th>
      <th>BestDeadliftKg</th>
      <th>TotalKg</th>
      <th>Place</th>
      <th>Wilks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Angie Belk Terry</td>
      <td>F</td>
      <td>Wraps</td>
      <td>47.0</td>
      <td>Mst 45-49</td>
      <td>59.60</td>
      <td>60</td>
      <td>NaN</td>
      <td>47.63</td>
      <td>NaN</td>
      <td>20.41</td>
      <td>NaN</td>
      <td>70.31</td>
      <td>138.35</td>
      <td>1</td>
      <td>155.05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Dawn Bogart</td>
      <td>F</td>
      <td>Single-ply</td>
      <td>42.0</td>
      <td>Mst 40-44</td>
      <td>58.51</td>
      <td>60</td>
      <td>NaN</td>
      <td>142.88</td>
      <td>NaN</td>
      <td>95.25</td>
      <td>NaN</td>
      <td>163.29</td>
      <td>401.42</td>
      <td>1</td>
      <td>456.38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Dawn Bogart</td>
      <td>F</td>
      <td>Single-ply</td>
      <td>42.0</td>
      <td>Open Senior</td>
      <td>58.51</td>
      <td>60</td>
      <td>NaN</td>
      <td>142.88</td>
      <td>NaN</td>
      <td>95.25</td>
      <td>NaN</td>
      <td>163.29</td>
      <td>401.42</td>
      <td>1</td>
      <td>456.38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Dawn Bogart</td>
      <td>F</td>
      <td>Raw</td>
      <td>42.0</td>
      <td>Open Senior</td>
      <td>58.51</td>
      <td>60</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>95.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>95.25</td>
      <td>1</td>
      <td>108.29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Destiny Dula</td>
      <td>F</td>
      <td>Raw</td>
      <td>18.0</td>
      <td>Teen 18-19</td>
      <td>63.68</td>
      <td>67.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.75</td>
      <td>NaN</td>
      <td>90.72</td>
      <td>122.47</td>
      <td>1</td>
      <td>130.47</td>
    </tr>
  </tbody>
</table>
</div>




```python
powerlifting_competitors.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 386414 entries, 0 to 386413
    Data columns (total 17 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   MeetID          386414 non-null  int64  
     1   Name            386414 non-null  object 
     2   Sex             386414 non-null  object 
     3   Equipment       386414 non-null  object 
     4   Age             147147 non-null  float64
     5   Division        370571 non-null  object 
     6   BodyweightKg    384012 non-null  float64
     7   WeightClassKg   382602 non-null  object 
     8   Squat4Kg        1243 non-null    float64
     9   BestSquatKg     298071 non-null  float64
     10  Bench4Kg        1962 non-null    float64
     11  BestBenchKg     356364 non-null  float64
     12  Deadlift4Kg     2800 non-null    float64
     13  BestDeadliftKg  317847 non-null  float64
     14  TotalKg         363237 non-null  float64
     15  Place           385322 non-null  object 
     16  Wilks           362194 non-null  float64
    dtypes: float64(10), int64(1), object(6)
    memory usage: 50.1+ MB



```python
powerlifting_competitors.describe()
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
      <th>MeetID</th>
      <th>Age</th>
      <th>BodyweightKg</th>
      <th>Squat4Kg</th>
      <th>BestSquatKg</th>
      <th>Bench4Kg</th>
      <th>BestBenchKg</th>
      <th>Deadlift4Kg</th>
      <th>BestDeadliftKg</th>
      <th>TotalKg</th>
      <th>Wilks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>386414.000000</td>
      <td>147147.000000</td>
      <td>384012.000000</td>
      <td>1243.000000</td>
      <td>298071.000000</td>
      <td>1962.000000</td>
      <td>356364.000000</td>
      <td>2800.000000</td>
      <td>317847.000000</td>
      <td>363237.000000</td>
      <td>362194.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5143.015804</td>
      <td>31.668237</td>
      <td>86.934912</td>
      <td>107.036404</td>
      <td>176.569941</td>
      <td>45.722905</td>
      <td>118.347509</td>
      <td>113.597193</td>
      <td>195.040633</td>
      <td>424.000249</td>
      <td>301.080601</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2552.099838</td>
      <td>12.900342</td>
      <td>23.140843</td>
      <td>166.976620</td>
      <td>69.222785</td>
      <td>151.668221</td>
      <td>54.848850</td>
      <td>170.201657</td>
      <td>61.580675</td>
      <td>196.355147</td>
      <td>116.360396</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>15.880000</td>
      <td>-440.500000</td>
      <td>-477.500000</td>
      <td>-360.000000</td>
      <td>-522.500000</td>
      <td>-461.000000</td>
      <td>-410.000000</td>
      <td>11.000000</td>
      <td>13.730000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2979.000000</td>
      <td>22.000000</td>
      <td>70.300000</td>
      <td>87.500000</td>
      <td>127.500000</td>
      <td>-90.000000</td>
      <td>79.380000</td>
      <td>110.000000</td>
      <td>147.500000</td>
      <td>272.160000</td>
      <td>237.380000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5960.000000</td>
      <td>28.000000</td>
      <td>83.200000</td>
      <td>145.000000</td>
      <td>174.630000</td>
      <td>90.250000</td>
      <td>115.000000</td>
      <td>157.500000</td>
      <td>195.000000</td>
      <td>424.110000</td>
      <td>319.660000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7175.000000</td>
      <td>39.000000</td>
      <td>100.000000</td>
      <td>212.500000</td>
      <td>217.720000</td>
      <td>167.500000</td>
      <td>150.000000</td>
      <td>219.990000</td>
      <td>238.140000</td>
      <td>565.000000</td>
      <td>379.290000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8481.000000</td>
      <td>95.000000</td>
      <td>242.400000</td>
      <td>450.000000</td>
      <td>573.790000</td>
      <td>378.750000</td>
      <td>488.500000</td>
      <td>418.000000</td>
      <td>460.400000</td>
      <td>1365.310000</td>
      <td>779.380000</td>
    </tr>
  </tbody>
</table>
</div>



Both tables include references to a `MeetID`, a unique key for each meet (competition) included in the database. Using this, generate a dataset combining the two tables into one.


```python
left = powerlifting_meets.set_index('MeetID')
right = powerlifting_competitors.set_index('MeetID')
```


```python
powerlifting_combined = left.join(right)

# Check your answer
q4.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
powerlifting_combined.head()
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
      <th>MeetPath</th>
      <th>Federation</th>
      <th>Date</th>
      <th>MeetCountry</th>
      <th>MeetState</th>
      <th>MeetTown</th>
      <th>MeetName</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Equipment</th>
      <th>...</th>
      <th>WeightClassKg</th>
      <th>Squat4Kg</th>
      <th>BestSquatKg</th>
      <th>Bench4Kg</th>
      <th>BestBenchKg</th>
      <th>Deadlift4Kg</th>
      <th>BestDeadliftKg</th>
      <th>TotalKg</th>
      <th>Place</th>
      <th>Wilks</th>
    </tr>
    <tr>
      <th>MeetID</th>
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
      <th>0</th>
      <td>365strong/1601</td>
      <td>365Strong</td>
      <td>2016-10-29</td>
      <td>USA</td>
      <td>NC</td>
      <td>Charlotte</td>
      <td>2016 Junior &amp; Senior National Powerlifting Cha...</td>
      <td>Angie Belk Terry</td>
      <td>F</td>
      <td>Wraps</td>
      <td>...</td>
      <td>60</td>
      <td>NaN</td>
      <td>47.63</td>
      <td>NaN</td>
      <td>20.41</td>
      <td>NaN</td>
      <td>70.31</td>
      <td>138.35</td>
      <td>1</td>
      <td>155.05</td>
    </tr>
    <tr>
      <th>0</th>
      <td>365strong/1601</td>
      <td>365Strong</td>
      <td>2016-10-29</td>
      <td>USA</td>
      <td>NC</td>
      <td>Charlotte</td>
      <td>2016 Junior &amp; Senior National Powerlifting Cha...</td>
      <td>Dawn Bogart</td>
      <td>F</td>
      <td>Single-ply</td>
      <td>...</td>
      <td>60</td>
      <td>NaN</td>
      <td>142.88</td>
      <td>NaN</td>
      <td>95.25</td>
      <td>NaN</td>
      <td>163.29</td>
      <td>401.42</td>
      <td>1</td>
      <td>456.38</td>
    </tr>
    <tr>
      <th>0</th>
      <td>365strong/1601</td>
      <td>365Strong</td>
      <td>2016-10-29</td>
      <td>USA</td>
      <td>NC</td>
      <td>Charlotte</td>
      <td>2016 Junior &amp; Senior National Powerlifting Cha...</td>
      <td>Dawn Bogart</td>
      <td>F</td>
      <td>Single-ply</td>
      <td>...</td>
      <td>60</td>
      <td>NaN</td>
      <td>142.88</td>
      <td>NaN</td>
      <td>95.25</td>
      <td>NaN</td>
      <td>163.29</td>
      <td>401.42</td>
      <td>1</td>
      <td>456.38</td>
    </tr>
    <tr>
      <th>0</th>
      <td>365strong/1601</td>
      <td>365Strong</td>
      <td>2016-10-29</td>
      <td>USA</td>
      <td>NC</td>
      <td>Charlotte</td>
      <td>2016 Junior &amp; Senior National Powerlifting Cha...</td>
      <td>Dawn Bogart</td>
      <td>F</td>
      <td>Raw</td>
      <td>...</td>
      <td>60</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>95.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>95.25</td>
      <td>1</td>
      <td>108.29</td>
    </tr>
    <tr>
      <th>0</th>
      <td>365strong/1601</td>
      <td>365Strong</td>
      <td>2016-10-29</td>
      <td>USA</td>
      <td>NC</td>
      <td>Charlotte</td>
      <td>2016 Junior &amp; Senior National Powerlifting Cha...</td>
      <td>Destiny Dula</td>
      <td>F</td>
      <td>Raw</td>
      <td>...</td>
      <td>67.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.75</td>
      <td>NaN</td>
      <td>90.72</td>
      <td>122.47</td>
      <td>1</td>
      <td>130.47</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
powerlifting_combined.tail()
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
      <th>MeetPath</th>
      <th>Federation</th>
      <th>Date</th>
      <th>MeetCountry</th>
      <th>MeetState</th>
      <th>MeetTown</th>
      <th>MeetName</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Equipment</th>
      <th>...</th>
      <th>WeightClassKg</th>
      <th>Squat4Kg</th>
      <th>BestSquatKg</th>
      <th>Bench4Kg</th>
      <th>BestBenchKg</th>
      <th>Deadlift4Kg</th>
      <th>BestDeadliftKg</th>
      <th>TotalKg</th>
      <th>Place</th>
      <th>Wilks</th>
    </tr>
    <tr>
      <th>MeetID</th>
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
      <th>8481</th>
      <td>xpc/2017-finals</td>
      <td>XPC</td>
      <td>2017-03-03</td>
      <td>USA</td>
      <td>OH</td>
      <td>Columbus</td>
      <td>2017 XPC Finals</td>
      <td>William Barabas</td>
      <td>M</td>
      <td>Multi-ply</td>
      <td>...</td>
      <td>125</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>347.5</td>
      <td>347.5</td>
      <td>2</td>
      <td>202.60</td>
    </tr>
    <tr>
      <th>8481</th>
      <td>xpc/2017-finals</td>
      <td>XPC</td>
      <td>2017-03-03</td>
      <td>USA</td>
      <td>OH</td>
      <td>Columbus</td>
      <td>2017 XPC Finals</td>
      <td>Justin Zottl</td>
      <td>M</td>
      <td>Multi-ply</td>
      <td>...</td>
      <td>125</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>322.5</td>
      <td>322.5</td>
      <td>3</td>
      <td>185.77</td>
    </tr>
    <tr>
      <th>8481</th>
      <td>xpc/2017-finals</td>
      <td>XPC</td>
      <td>2017-03-03</td>
      <td>USA</td>
      <td>OH</td>
      <td>Columbus</td>
      <td>2017 XPC Finals</td>
      <td>Jake Anderson</td>
      <td>M</td>
      <td>Multi-ply</td>
      <td>...</td>
      <td>125</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>367.5</td>
      <td>367.5</td>
      <td>1</td>
      <td>211.17</td>
    </tr>
    <tr>
      <th>8481</th>
      <td>xpc/2017-finals</td>
      <td>XPC</td>
      <td>2017-03-03</td>
      <td>USA</td>
      <td>OH</td>
      <td>Columbus</td>
      <td>2017 XPC Finals</td>
      <td>Jeff Bumanglag</td>
      <td>M</td>
      <td>Multi-ply</td>
      <td>...</td>
      <td>140</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>320.0</td>
      <td>320.0</td>
      <td>3</td>
      <td>181.85</td>
    </tr>
    <tr>
      <th>8481</th>
      <td>xpc/2017-finals</td>
      <td>XPC</td>
      <td>2017-03-03</td>
      <td>USA</td>
      <td>OH</td>
      <td>Columbus</td>
      <td>2017 XPC Finals</td>
      <td>Shane Hammock</td>
      <td>M</td>
      <td>Multi-ply</td>
      <td>...</td>
      <td>140</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>362.5</td>
      <td>362.5</td>
      <td>2</td>
      <td>205.18</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
powerlifting_combined.columns
```




    Index(['MeetPath', 'Federation', 'Date', 'MeetCountry', 'MeetState',
           'MeetTown', 'MeetName', 'Name', 'Sex', 'Equipment', 'Age', 'Division',
           'BodyweightKg', 'WeightClassKg', 'Squat4Kg', 'BestSquatKg', 'Bench4Kg',
           'BestBenchKg', 'Deadlift4Kg', 'BestDeadliftKg', 'TotalKg', 'Place',
           'Wilks'],
          dtype='object')




```python
powerlifting_combined.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 386414 entries, 0 to 8481
    Data columns (total 23 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   MeetPath        386414 non-null  object 
     1   Federation      386414 non-null  object 
     2   Date            386414 non-null  object 
     3   MeetCountry     386414 non-null  object 
     4   MeetState       314271 non-null  object 
     5   MeetTown        292414 non-null  object 
     6   MeetName        386414 non-null  object 
     7   Name            386414 non-null  object 
     8   Sex             386414 non-null  object 
     9   Equipment       386414 non-null  object 
     10  Age             147147 non-null  float64
     11  Division        370571 non-null  object 
     12  BodyweightKg    384012 non-null  float64
     13  WeightClassKg   382602 non-null  object 
     14  Squat4Kg        1243 non-null    float64
     15  BestSquatKg     298071 non-null  float64
     16  Bench4Kg        1962 non-null    float64
     17  BestBenchKg     356364 non-null  float64
     18  Deadlift4Kg     2800 non-null    float64
     19  BestDeadliftKg  317847 non-null  float64
     20  TotalKg         363237 non-null  float64
     21  Place           385322 non-null  object 
     22  Wilks           362194 non-null  float64
    dtypes: float64(10), object(13)
    memory usage: 70.8+ MB



```python
#q4.hint()
#q4.solution()
```

# Congratulations!

You've finished the Pandas micro-course.  Many data scientists feel efficiency with Pandas is the most useful and practical skill they have, because it allows you to progress quickly in any project you have.

If you'd like to apply your new skills to examining geospatial data, you're encouraged to check out our **[Geospatial Analysis](https://www.kaggle.com/learn/geospatial-analysis)** micro-course.

You can also take advantage of your Pandas skills by entering a **[Kaggle Competition](https://www.kaggle.com/competitions)** or by answering a question you find interesting using **[Kaggle Datasets](https://www.kaggle.com/datasets)**.

---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/pandas/discussion) to chat with other learners.*

{% if page.comments != false %}
{% include disqus.html %}
{% endif %}
