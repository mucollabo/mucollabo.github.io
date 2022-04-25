---
classes: wide

title: "Pandas Data-types and Missing"

excerpt: "Run the following cell to load your data and some utility functions."

categories:
- kaggle

tags:
- python
- pandas
- data

---


**This notebook is an exercise in the [Pandas](https://www.kaggle.com/learn/pandas) course.  You can reference the tutorial at [this link](https://www.kaggle.com/residentmario/data-types-and-missing-values).**

---


# Introduction

Run the following cell to load your data and some utility functions.


```python
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.data_types_and_missing_data import *
print("Setup complete.")
```

    Setup complete.


# Exercises

## 1. 
What is the data type of the `points` column in the dataset?


```python
# Your code here
dtype = reviews.points.dtype

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
Create a Series from entries in the `points` column, but convert the entries to strings. Hint: strings are `str` in native Python.


```python
point_strings = reviews.points.astype('str')

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
Sometimes the price column is null. How many reviews in the dataset are missing a price?


```python
reviews[pd.isnull(reviews.price)]
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
      <th>13</th>
      <td>Italy</td>
      <td>This is dominated by oak and oak-driven aromas...</td>
      <td>Rosso</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Masseria Setteporte 2012 Rosso  (Etna)</td>
      <td>Nerello Mascalese</td>
      <td>Masseria Setteporte</td>
    </tr>
    <tr>
      <th>30</th>
      <td>France</td>
      <td>Red cherry fruit comes laced with light tannin...</td>
      <td>Nouveau</td>
      <td>86</td>
      <td>NaN</td>
      <td>Beaujolais</td>
      <td>Beaujolais-Villages</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine de la Madone 2012 Nouveau  (Beaujolais...</td>
      <td>Gamay</td>
      <td>Domaine de la Madone</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Italy</td>
      <td>Merlot and Nero d'Avola form the base for this...</td>
      <td>Calanìca Nero d'Avola-Merlot</td>
      <td>86</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Duca di Salaparuta 2010 Calanìca Nero d'Avola-...</td>
      <td>Red Blend</td>
      <td>Duca di Salaparuta</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Italy</td>
      <td>Part of the extended Calanìca series, this Gri...</td>
      <td>Calanìca Grillo-Viognier</td>
      <td>86</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Sicilia</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Duca di Salaparuta 2011 Calanìca Grillo-Viogni...</td>
      <td>White Blend</td>
      <td>Duca di Salaparuta</td>
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
      <th>129844</th>
      <td>Italy</td>
      <td>Doga delle Clavule is a neutral, mineral-drive...</td>
      <td>Doga delle Clavule</td>
      <td>86</td>
      <td>NaN</td>
      <td>Tuscany</td>
      <td>Morellino di Scansano</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Caparzo 2006 Doga delle Clavule  (Morellino di...</td>
      <td>Sangiovese</td>
      <td>Caparzo</td>
    </tr>
    <tr>
      <th>129860</th>
      <td>Portugal</td>
      <td>This rich wine has a firm structure as well as...</td>
      <td>Pacheca Superior</td>
      <td>90</td>
      <td>NaN</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta da Pacheca 2013 Pacheca Superior Red (D...</td>
      <td>Portuguese Red</td>
      <td>Quinta da Pacheca</td>
    </tr>
    <tr>
      <th>129863</th>
      <td>Portugal</td>
      <td>This mature wine that has 50% Touriga Nacional...</td>
      <td>Reserva</td>
      <td>90</td>
      <td>NaN</td>
      <td>Dão</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Seacampo 2011 Reserva Red (Dão)</td>
      <td>Portuguese Red</td>
      <td>Seacampo</td>
    </tr>
    <tr>
      <th>129893</th>
      <td>Italy</td>
      <td>Aromas of passion fruit, hay and a vegetal not...</td>
      <td>Corte Menini</td>
      <td>91</td>
      <td>NaN</td>
      <td>Veneto</td>
      <td>Soave Classico</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Le Mandolare 2015 Corte Menini  (Soave Classico)</td>
      <td>Garganega</td>
      <td>Le Mandolare</td>
    </tr>
    <tr>
      <th>129964</th>
      <td>France</td>
      <td>Initially quite muted, this wine slowly develo...</td>
      <td>Domaine Saint-Rémy Herrenweg</td>
      <td>90</td>
      <td>NaN</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Ehrhart 2013 Domaine Saint-Rémy Herren...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Ehrhart</td>
    </tr>
  </tbody>
</table>
<p>8996 rows × 13 columns</p>
</div>




```python
type(reviews[pd.isnull(reviews.price)])
```




    pandas.core.frame.DataFrame




```python
n_missing_prices = len(reviews[pd.isnull(reviews.price)])

# Check your answer
q3.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
n_missing_prices
```




    8996




```python
q3.hint()
#q3.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#3366cc">Hint:</span> Use `pd.isnull()`.


## 4.
What are the most common wine-producing regions? Create a Series counting the number of times each value occurs in the `region_1` field. This field is often missing data, so replace missing values with `Unknown`. Sort in descending order.  Your output should look something like this:

```
Unknown                    21247
Napa Valley                 4480
                           ...  
Bardolino Superiore            1
Primitivo del Tarantino        1
Name: region_1, Length: 1230, dtype: int64
```


```python
reviews.region_1.fillna('Unknown').value_counts()
```




    Unknown                    21247
    Napa Valley                 4480
    Columbia Valley (WA)        4124
    Russian River Valley        3091
    California                  2629
                               ...  
    Lamezia                        1
    Trentino Superiore             1
    Grave del Friuli               1
    Vin Santo di Carmignano        1
    Paestum                        1
    Name: region_1, Length: 1230, dtype: int64




```python
reviews_per_region = reviews.region_1.fillna('Unknown').value_counts()

# Check your answer
q4.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
#q4.hint()
#q4.solution()
```

# Keep going

Move on to **[renaming and combining](https://www.kaggle.com/residentmario/renaming-and-combining)**.

---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/pandas/discussion) to chat with other learners.*
