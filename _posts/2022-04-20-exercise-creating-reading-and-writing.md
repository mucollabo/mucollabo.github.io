---
classes: wide

title: "Pandas Creating, Reading and Writing "

excerpt: "The first step in most data analytics projects is reading the data file."

categories:
- kaggle

tags:
- python
- pandas
- data

---

**This notebook is an exercise in the [Pandas](https://www.kaggle.com/learn/pandas) course.  You can reference the tutorial at [this link](https://www.kaggle.com/residentmario/creating-reading-and-writing).**

---


# Introduction

The first step in most data analytics projects is reading the data file. In this exercise, you'll create Series and DataFrame objects, both by hand and by reading data files.

Run the code cell below to load libraries you will need (including code to check your answers).


```python
import pandas as pd
pd.set_option('max_rows', 5)
from learntools.core import binder; binder.bind(globals())
from learntools.pandas.creating_reading_and_writing import *
print("Setup complete.")
```

    Setup complete.


# Exercises

## 1.

In the cell below, create a DataFrame `fruits` that looks like this:

![](https://i.imgur.com/Ax3pp2A.png)


```python
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
fruits = pd.DataFrame({'Apples':[30], 'Bananas':[21]})

# Check your answer
q1.check()
fruits
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
      <th>Apples</th>
      <th>Bananas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
#q1.hint()
#q1.solution()
```

## 2.

Create a dataframe `fruit_sales` that matches the diagram below:

![](https://i.imgur.com/CHPn7ZF.png)


```python
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = pd.DataFrame({'Apples':[35, 41], 'Bananas':[21, 34]}, index=['2017 Sales', '2018 Sales'])

# Check your answer
q2.check()
fruit_sales
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
      <th>Apples</th>
      <th>Bananas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017 Sales</th>
      <td>35</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2018 Sales</th>
      <td>41</td>
      <td>34</td>
    </tr>
  </tbody>
</table>
</div>




```python
#q2.hint()
#q2.solution()
```

## 3.

Create a variable `ingredients` with a Series that looks like:

```
Flour     4 cups
Milk       1 cup
Eggs     2 large
Spam       1 can
Name: Dinner, dtype: object
```


```python
ingredients = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')

# Check your answer
q3.check()
ingredients
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>





    Flour     4 cups
    Milk       1 cup
    Eggs     2 large
    Spam       1 can
    Name: Dinner, dtype: object




```python
#q3.hint()
#q3.solution()
```

## 4.

Read the following csv dataset of wine reviews into a DataFrame called `reviews`:

![](https://i.imgur.com/74RCZtU.png)

The filepath to the csv file is `../input/wine-reviews/winemag-data_first150k.csv`. The first few lines look like:

```
,country,description,designation,points,price,province,region_1,region_2,variety,winery
0,US,"This tremendous 100% varietal wine[...]",Martha's Vineyard,96,235.0,California,Napa Valley,Napa,Cabernet Sauvignon,Heitz
1,Spain,"Ripe aromas of fig, blackberry and[...]",Carodorum Selección Especial Reserva,96,110.0,Northern Spain,Toro,,Tinta de Toro,Bodega Carmen Rodríguez
```


```python
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

# Check your answer
q4.check()
reviews
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
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US</td>
      <td>This tremendous 100% varietal wine hails from ...</td>
      <td>Martha's Vineyard</td>
      <td>96</td>
      <td>235.0</td>
      <td>California</td>
      <td>Napa Valley</td>
      <td>Napa</td>
      <td>Cabernet Sauvignon</td>
      <td>Heitz</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spain</td>
      <td>Ripe aromas of fig, blackberry and cassis are ...</td>
      <td>Carodorum Selección Especial Reserva</td>
      <td>96</td>
      <td>110.0</td>
      <td>Northern Spain</td>
      <td>Toro</td>
      <td>NaN</td>
      <td>Tinta de Toro</td>
      <td>Bodega Carmen Rodríguez</td>
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
    </tr>
    <tr>
      <th>150928</th>
      <td>France</td>
      <td>A perfect salmon shade, with scents of peaches...</td>
      <td>Grand Brut Rosé</td>
      <td>90</td>
      <td>52.0</td>
      <td>Champagne</td>
      <td>Champagne</td>
      <td>NaN</td>
      <td>Champagne Blend</td>
      <td>Gosset</td>
    </tr>
    <tr>
      <th>150929</th>
      <td>Italy</td>
      <td>More Pinot Grigios should taste like this. A r...</td>
      <td>NaN</td>
      <td>90</td>
      <td>15.0</td>
      <td>Northeastern Italy</td>
      <td>Alto Adige</td>
      <td>NaN</td>
      <td>Pinot Grigio</td>
      <td>Alois Lageder</td>
    </tr>
  </tbody>
</table>
<p>150930 rows × 10 columns</p>
</div>




```python
q4.hint()
#q4.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#3366cc">Hint:</span> Note that the csv file begins with an unnamed column of increasing integers. We want this to be used as the index. Check out the description of the `index_col` keyword argument in [the docs for `read_csv`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html).


## 5.

Run the cell below to create and display a DataFrame called `animals`:


```python
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals
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
      <th>Cows</th>
      <th>Goats</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Year 1</th>
      <td>12</td>
      <td>22</td>
    </tr>
    <tr>
      <th>Year 2</th>
      <td>20</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>



In the cell below, write code to save this DataFrame to disk as a csv file with the name `cows_and_goats.csv`.


```python
# Your code goes here
animals.to_csv("cows_and_goats.csv")
# Check your answer
q5.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
#q5.hint()
#q5.solution()
```

# Keep going

Move on to learn about **[indexing, selecting and assigning](https://www.kaggle.com/residentmario/indexing-selecting-assigning)**.

---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/pandas/discussion) to chat with other learners.*
