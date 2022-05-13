**This notebook is an exercise in the [Data Visualization](https://www.kaggle.com/learn/data-visualization) course.  You can reference the tutorial at [this link](https://www.kaggle.com/alexisbcook/distributions).**

---


In this exercise, you will use your new knowledge to propose a solution to a real-world scenario.  To succeed, you will need to import data into Python, answer questions using the data, and generate **histograms** and **density plots** to understand patterns in the data.

## Scenario

You'll work with a real-world dataset containing information collected from microscopic images of breast cancer tumors, similar to the image below.

![ex4_cancer_image](https://i.imgur.com/qUESsJe.png)

Each tumor has been labeled as either [**benign**](https://en.wikipedia.org/wiki/Benign_tumor) (_noncancerous_) or **malignant** (_cancerous_).

To learn more about how this kind of data is used to create intelligent algorithms to classify tumors in medical settings, **watch the short video [at this link](https://www.youtube.com/watch?v=9Mz84cwVmS0)**!



## Setup

Run the next cell to import and configure the Python libraries that you need to complete the exercise.


```python
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
```

    Setup Complete


The questions below will give you feedback on your work. Run the following cell to set up our feedback system.


```python
# Set up code checking
import os
if not os.path.exists("../input/cancer_b.csv"):
    os.symlink("../input/data-for-datavis/cancer_b.csv", "../input/cancer_b.csv")
    os.symlink("../input/data-for-datavis/cancer_m.csv", "../input/cancer_m.csv")
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex5 import *
print("Setup Complete")
```

    Setup Complete


## Step 1: Load the data

In this step, you will load two data files.
- Load the data file corresponding to **benign** tumors into a DataFrame called `cancer_b_data`.  The corresponding filepath is `cancer_b_filepath`.  Use the `"Id"` column to label the rows.
- Load the data file corresponding to **malignant** tumors into a DataFrame called `cancer_m_data`.  The corresponding filepath is `cancer_m_filepath`.  Use the `"Id"` column to label the rows.


```python
# Paths of the files to read
cancer_b_filepath = "../input/cancer_b.csv"
cancer_m_filepath = "../input/cancer_m.csv"

# Fill in the line below to read the (benign) file into a variable cancer_b_data
cancer_b_data = pd.read_csv(cancer_b_filepath, index_col="Id")

# Fill in the line below to read the (malignant) file into a variable cancer_m_data
cancer_m_data = pd.read_csv(cancer_m_filepath, index_col="Id")

# Run the line below with no changes to check that you've loaded the data correctly
step_1.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# Lines below will give you a hint or solution code
#step_1.hint()
#step_1.solution()
```

## Step 2: Review the data

Use a Python command to print the first 5 rows of the data for benign tumors.


```python
# Print the first five rows of the (benign) data
cancer_b_data.head()
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
      <th>Diagnosis</th>
      <th>Radius (mean)</th>
      <th>Texture (mean)</th>
      <th>Perimeter (mean)</th>
      <th>Area (mean)</th>
      <th>Smoothness (mean)</th>
      <th>Compactness (mean)</th>
      <th>Concavity (mean)</th>
      <th>Concave points (mean)</th>
      <th>Symmetry (mean)</th>
      <th>...</th>
      <th>Radius (worst)</th>
      <th>Texture (worst)</th>
      <th>Perimeter (worst)</th>
      <th>Area (worst)</th>
      <th>Smoothness (worst)</th>
      <th>Compactness (worst)</th>
      <th>Concavity (worst)</th>
      <th>Concave points (worst)</th>
      <th>Symmetry (worst)</th>
      <th>Fractal dimension (worst)</th>
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
      <th>8510426</th>
      <td>B</td>
      <td>13.540</td>
      <td>14.36</td>
      <td>87.46</td>
      <td>566.3</td>
      <td>0.09779</td>
      <td>0.08129</td>
      <td>0.06664</td>
      <td>0.047810</td>
      <td>0.1885</td>
      <td>...</td>
      <td>15.110</td>
      <td>19.26</td>
      <td>99.70</td>
      <td>711.2</td>
      <td>0.14400</td>
      <td>0.17730</td>
      <td>0.23900</td>
      <td>0.12880</td>
      <td>0.2977</td>
      <td>0.07259</td>
    </tr>
    <tr>
      <th>8510653</th>
      <td>B</td>
      <td>13.080</td>
      <td>15.71</td>
      <td>85.63</td>
      <td>520.0</td>
      <td>0.10750</td>
      <td>0.12700</td>
      <td>0.04568</td>
      <td>0.031100</td>
      <td>0.1967</td>
      <td>...</td>
      <td>14.500</td>
      <td>20.49</td>
      <td>96.09</td>
      <td>630.5</td>
      <td>0.13120</td>
      <td>0.27760</td>
      <td>0.18900</td>
      <td>0.07283</td>
      <td>0.3184</td>
      <td>0.08183</td>
    </tr>
    <tr>
      <th>8510824</th>
      <td>B</td>
      <td>9.504</td>
      <td>12.44</td>
      <td>60.34</td>
      <td>273.9</td>
      <td>0.10240</td>
      <td>0.06492</td>
      <td>0.02956</td>
      <td>0.020760</td>
      <td>0.1815</td>
      <td>...</td>
      <td>10.230</td>
      <td>15.66</td>
      <td>65.13</td>
      <td>314.9</td>
      <td>0.13240</td>
      <td>0.11480</td>
      <td>0.08867</td>
      <td>0.06227</td>
      <td>0.2450</td>
      <td>0.07773</td>
    </tr>
    <tr>
      <th>854941</th>
      <td>B</td>
      <td>13.030</td>
      <td>18.42</td>
      <td>82.61</td>
      <td>523.8</td>
      <td>0.08983</td>
      <td>0.03766</td>
      <td>0.02562</td>
      <td>0.029230</td>
      <td>0.1467</td>
      <td>...</td>
      <td>13.300</td>
      <td>22.81</td>
      <td>84.46</td>
      <td>545.9</td>
      <td>0.09701</td>
      <td>0.04619</td>
      <td>0.04833</td>
      <td>0.05013</td>
      <td>0.1987</td>
      <td>0.06169</td>
    </tr>
    <tr>
      <th>85713702</th>
      <td>B</td>
      <td>8.196</td>
      <td>16.84</td>
      <td>51.71</td>
      <td>201.9</td>
      <td>0.08600</td>
      <td>0.05943</td>
      <td>0.01588</td>
      <td>0.005917</td>
      <td>0.1769</td>
      <td>...</td>
      <td>8.964</td>
      <td>21.96</td>
      <td>57.26</td>
      <td>242.2</td>
      <td>0.12970</td>
      <td>0.13570</td>
      <td>0.06880</td>
      <td>0.02564</td>
      <td>0.3105</td>
      <td>0.07409</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
cancer_b_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 357 entries, 8510426 to 92751
    Data columns (total 31 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   Diagnosis                  357 non-null    object 
     1   Radius (mean)              357 non-null    float64
     2   Texture (mean)             357 non-null    float64
     3   Perimeter (mean)           357 non-null    float64
     4   Area (mean)                357 non-null    float64
     5   Smoothness (mean)          357 non-null    float64
     6   Compactness (mean)         357 non-null    float64
     7   Concavity (mean)           357 non-null    float64
     8   Concave points (mean)      357 non-null    float64
     9   Symmetry (mean)            357 non-null    float64
     10  Fractal dimension (mean)   357 non-null    float64
     11  Radius (se)                357 non-null    float64
     12  Texture (se)               357 non-null    float64
     13  Perimeter (se)             357 non-null    float64
     14  Area (se)                  357 non-null    float64
     15  Smoothness (se)            357 non-null    float64
     16  Compactness (se)           357 non-null    float64
     17  Concavity (se)             357 non-null    float64
     18  Concave points (se)        357 non-null    float64
     19  Symmetry (se)              357 non-null    float64
     20  Fractal dimension (se)     357 non-null    float64
     21  Radius (worst)             357 non-null    float64
     22  Texture (worst)            357 non-null    float64
     23  Perimeter (worst)          357 non-null    float64
     24  Area (worst)               357 non-null    float64
     25  Smoothness (worst)         357 non-null    float64
     26  Compactness (worst)        357 non-null    float64
     27  Concavity (worst)          357 non-null    float64
     28  Concave points (worst)     357 non-null    float64
     29  Symmetry (worst)           357 non-null    float64
     30  Fractal dimension (worst)  357 non-null    float64
    dtypes: float64(30), object(1)
    memory usage: 89.2+ KB


Use a Python command to print the first 5 rows of the data for malignant tumors.


```python
# Print the first five rows of the (malignant) data
cancer_m_data.head()
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
      <th>Diagnosis</th>
      <th>Radius (mean)</th>
      <th>Texture (mean)</th>
      <th>Perimeter (mean)</th>
      <th>Area (mean)</th>
      <th>Smoothness (mean)</th>
      <th>Compactness (mean)</th>
      <th>Concavity (mean)</th>
      <th>Concave points (mean)</th>
      <th>Symmetry (mean)</th>
      <th>...</th>
      <th>Radius (worst)</th>
      <th>Texture (worst)</th>
      <th>Perimeter (worst)</th>
      <th>Area (worst)</th>
      <th>Smoothness (worst)</th>
      <th>Compactness (worst)</th>
      <th>Concavity (worst)</th>
      <th>Concave points (worst)</th>
      <th>Symmetry (worst)</th>
      <th>Fractal dimension (worst)</th>
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
      <th>842302</th>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>842517</th>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>84300903</th>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>84348301</th>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>84358402</th>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
cancer_m_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 212 entries, 842302 to 927241
    Data columns (total 31 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   Diagnosis                  212 non-null    object 
     1   Radius (mean)              212 non-null    float64
     2   Texture (mean)             212 non-null    float64
     3   Perimeter (mean)           212 non-null    float64
     4   Area (mean)                212 non-null    float64
     5   Smoothness (mean)          212 non-null    float64
     6   Compactness (mean)         212 non-null    float64
     7   Concavity (mean)           212 non-null    float64
     8   Concave points (mean)      212 non-null    float64
     9   Symmetry (mean)            212 non-null    float64
     10  Fractal dimension (mean)   212 non-null    float64
     11  Radius (se)                212 non-null    float64
     12  Texture (se)               212 non-null    float64
     13  Perimeter (se)             212 non-null    float64
     14  Area (se)                  212 non-null    float64
     15  Smoothness (se)            212 non-null    float64
     16  Compactness (se)           212 non-null    float64
     17  Concavity (se)             212 non-null    float64
     18  Concave points (se)        212 non-null    float64
     19  Symmetry (se)              212 non-null    float64
     20  Fractal dimension (se)     212 non-null    float64
     21  Radius (worst)             212 non-null    float64
     22  Texture (worst)            212 non-null    float64
     23  Perimeter (worst)          212 non-null    float64
     24  Area (worst)               212 non-null    float64
     25  Smoothness (worst)         212 non-null    float64
     26  Compactness (worst)        212 non-null    float64
     27  Concavity (worst)          212 non-null    float64
     28  Concave points (worst)     212 non-null    float64
     29  Symmetry (worst)           212 non-null    float64
     30  Fractal dimension (worst)  212 non-null    float64
    dtypes: float64(30), object(1)
    memory usage: 53.0+ KB


In the datasets, each row corresponds to a different image.  Each dataset has 31 different columns, corresponding to:
- 1 column (`'Diagnosis'`) that classifies tumors as either benign (which appears in the dataset as **`B`**) or malignant (__`M`__), and
- 30 columns containing different measurements collected from the images.

Use the first 5 rows of the data (for benign and malignant tumors) to answer the questions below.


```python
# Fill in the line below: In the first five rows of the data for benign tumors, what is the
# largest value for 'Perimeter (mean)'?
max_perim = 87.46

# Fill in the line below: What is the value for 'Radius (mean)' for the tumor with Id 842517?
mean_radius = 20.57

# Check your answers
step_2.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



```python
# Lines below will give you a hint or solution code
#step_2.hint()
#step_2.solution()
```

## Step 3: Investigating differences

#### Part A

Use the code cell below to create two histograms that show the distribution in values for `'Area (mean)'` for both benign and malignant tumors.  (_To permit easy comparison, create a single figure containing both histograms in the code cell below._)


```python
# Histograms for benign and maligant tumors
plt.figure(figsize=(14,6))
sns.distplot(a=cancer_b_data['Area (mean)'], kde=False, color="red") # Your code here (benign tumors)
sns.distplot(a=cancer_m_data['Area (mean)'], kde=False, color="darkblue") # Your code here (malignant tumors)


# Check your answer
step_3.a.check()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



    
![png](exercise-distributions_files/exercise-distributions_20_2.png)
    



```python
# Lines below will give you a hint or solution code
#step_3.a.hint()
#step_3.a.solution_plot()
```

#### Part B

A researcher approaches you for help with identifying how the `'Area (mean)'` column can be used to understand the difference between benign and malignant tumors.  Based on the histograms above, 
- Do malignant tumors have higher or lower values for `'Area (mean)'` (relative to benign tumors), on average?
- Which tumor type seems to have a larger range of potential values?


```python
#step_3.b.hint()
```


```python
# Check your answer (Run this code cell to receive credit!)
step_3.b.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc99">Solution:</span> Malignant tumors have higher values for `'Area (mean)'`, on average. Malignant tumors have a larger range of potential values.


## Step 4: A very useful column

#### Part A

Use the code cell below to create two KDE plots that show the distribution in values for `'Radius (worst)'` for both benign and malignant tumors.  (_To permit easy comparison, create a single figure containing both KDE plots in the code cell below._)


```python
# KDE plots for benign and malignant tumors
plt.figure(figsize=(12, 6))
sns.kdeplot(data=cancer_b_data['Radius (worst)'], shade=True, color="darkblue") # Your code here (benign tumors)
sns.kdeplot(data=cancer_m_data['Radius (worst)'], shade=True, color="orange") # Your code here (malignant tumors)

# Check your answer
step_4.a.check()

plt.legend()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>





    <matplotlib.legend.Legend at 0x7f352cb28e90>




    
![png](exercise-distributions_files/exercise-distributions_26_3.png)
    



```python
# Lines below will give you a hint or solution code
#step_4.a.hint()
#step_4.a.solution_plot()
```

#### Part B

A hospital has recently started using an algorithm that can diagnose tumors with high accuracy.  Given a tumor with a value for `'Radius (worst)'` of 25, do you think the algorithm is more likely to classify the tumor as benign or malignant?


```python
#step_4.b.hint()
```


```python
# Check your answer (Run this code cell to receive credit!)
step_4.b.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc99">Solution:</span> The algorithm is more likely to classify the tumor as malignant. This is because the curve for malignant tumors is much higher than the curve for benign tumors around a value of 25 -- and an algorithm that gets high accuracy is likely to make decisions based on this pattern in the data.


## Keep going

Review all that you've learned and explore how to further customize your plots in the **[next tutorial](https://www.kaggle.com/alexisbcook/choosing-plot-types-and-custom-styles)**!

---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/data-visualization/discussion) to chat with other learners.*
