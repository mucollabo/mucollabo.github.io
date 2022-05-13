---
classes: wide

title: "Bar Charts and Heatmaps"

excerpt: "Use color or length to compare categories in a dataset"

categories:
- visualization
- barplot
- heatmap

---


**This notebook is an exercise in the [Data Visualization](https://www.kaggle.com/learn/data-visualization) course.  You can reference the tutorial at [this link](https://www.kaggle.com/alexisbcook/bar-charts-and-heatmaps).**

---


In this exercise, you will use your new knowledge to propose a solution to a real-world scenario.  To succeed, you will need to import data into Python, answer questions using the data, and generate **bar charts** and **heatmaps** to understand patterns in the data.

## Scenario

You've recently decided to create your very own video game!  As an avid reader of [IGN Game Reviews](https://www.ign.com/reviews/games), you hear about all of the most recent game releases, along with the ranking they've received from experts, ranging from 0 (_Disaster_) to 10 (_Masterpiece_).

![ex2_ign](https://i.imgur.com/Oh06Fu1.png)

You're interested in using [IGN reviews](https://www.ign.com/reviews/games) to guide the design of your upcoming game.  Thankfully, someone has summarized the rankings in a really useful CSV file that you can use to guide your analysis.

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
if not os.path.exists("../input/ign_scores.csv"):
    os.symlink("../input/data-for-datavis/ign_scores.csv", "../input/ign_scores.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex3 import *
print("Setup Complete")
```

    Setup Complete


## Step 1: Load the data

Read the IGN data file into `ign_data`.  Use the `"Platform"` column to label the rows.


```python
# Path of the file to read
ign_filepath = "../input/ign_scores.csv"

# Fill in the line below to read the file into a variable ign_data
ign_data = pd.read_csv(ign_filepath, index_col="Platform")

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

Use a Python command to print the entire dataset.


```python
# Print the data
ign_data
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
      <th>Action</th>
      <th>Action, Adventure</th>
      <th>Adventure</th>
      <th>Fighting</th>
      <th>Platformer</th>
      <th>Puzzle</th>
      <th>RPG</th>
      <th>Racing</th>
      <th>Shooter</th>
      <th>Simulation</th>
      <th>Sports</th>
      <th>Strategy</th>
    </tr>
    <tr>
      <th>Platform</th>
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
      <th>Dreamcast</th>
      <td>6.882857</td>
      <td>7.511111</td>
      <td>6.281818</td>
      <td>8.200000</td>
      <td>8.340000</td>
      <td>8.088889</td>
      <td>7.700000</td>
      <td>7.042500</td>
      <td>7.616667</td>
      <td>7.628571</td>
      <td>7.272222</td>
      <td>6.433333</td>
    </tr>
    <tr>
      <th>Game Boy Advance</th>
      <td>6.373077</td>
      <td>7.507692</td>
      <td>6.057143</td>
      <td>6.226316</td>
      <td>6.970588</td>
      <td>6.532143</td>
      <td>7.542857</td>
      <td>6.657143</td>
      <td>6.444444</td>
      <td>6.928571</td>
      <td>6.694444</td>
      <td>7.175000</td>
    </tr>
    <tr>
      <th>Game Boy Color</th>
      <td>6.272727</td>
      <td>8.166667</td>
      <td>5.307692</td>
      <td>4.500000</td>
      <td>6.352941</td>
      <td>6.583333</td>
      <td>7.285714</td>
      <td>5.897436</td>
      <td>4.500000</td>
      <td>5.900000</td>
      <td>5.790698</td>
      <td>7.400000</td>
    </tr>
    <tr>
      <th>GameCube</th>
      <td>6.532584</td>
      <td>7.608333</td>
      <td>6.753846</td>
      <td>7.422222</td>
      <td>6.665714</td>
      <td>6.133333</td>
      <td>7.890909</td>
      <td>6.852632</td>
      <td>6.981818</td>
      <td>8.028571</td>
      <td>7.481319</td>
      <td>7.116667</td>
    </tr>
    <tr>
      <th>Nintendo 3DS</th>
      <td>6.670833</td>
      <td>7.481818</td>
      <td>7.414286</td>
      <td>6.614286</td>
      <td>7.503448</td>
      <td>8.000000</td>
      <td>7.719231</td>
      <td>6.900000</td>
      <td>7.033333</td>
      <td>7.700000</td>
      <td>6.388889</td>
      <td>7.900000</td>
    </tr>
    <tr>
      <th>Nintendo 64</th>
      <td>6.649057</td>
      <td>8.250000</td>
      <td>7.000000</td>
      <td>5.681250</td>
      <td>6.889655</td>
      <td>7.461538</td>
      <td>6.050000</td>
      <td>6.939623</td>
      <td>8.042857</td>
      <td>5.675000</td>
      <td>6.967857</td>
      <td>6.900000</td>
    </tr>
    <tr>
      <th>Nintendo DS</th>
      <td>5.903608</td>
      <td>7.240000</td>
      <td>6.259804</td>
      <td>6.320000</td>
      <td>6.840000</td>
      <td>6.604615</td>
      <td>7.222619</td>
      <td>6.038636</td>
      <td>6.965217</td>
      <td>5.874359</td>
      <td>5.936667</td>
      <td>6.644737</td>
    </tr>
    <tr>
      <th>Nintendo DSi</th>
      <td>6.827027</td>
      <td>8.500000</td>
      <td>6.090909</td>
      <td>7.500000</td>
      <td>7.250000</td>
      <td>6.810526</td>
      <td>7.166667</td>
      <td>6.563636</td>
      <td>6.500000</td>
      <td>5.195652</td>
      <td>5.644444</td>
      <td>6.566667</td>
    </tr>
    <tr>
      <th>PC</th>
      <td>6.805791</td>
      <td>7.334746</td>
      <td>7.136798</td>
      <td>7.166667</td>
      <td>7.410938</td>
      <td>6.924706</td>
      <td>7.759930</td>
      <td>7.032418</td>
      <td>7.084878</td>
      <td>7.104889</td>
      <td>6.902424</td>
      <td>7.310207</td>
    </tr>
    <tr>
      <th>PlayStation</th>
      <td>6.016406</td>
      <td>7.933333</td>
      <td>6.313725</td>
      <td>6.553731</td>
      <td>6.579070</td>
      <td>6.757895</td>
      <td>7.910000</td>
      <td>6.773387</td>
      <td>6.424000</td>
      <td>6.918182</td>
      <td>6.751220</td>
      <td>6.496875</td>
    </tr>
    <tr>
      <th>PlayStation 2</th>
      <td>6.467361</td>
      <td>7.250000</td>
      <td>6.315152</td>
      <td>7.306349</td>
      <td>7.068421</td>
      <td>6.354545</td>
      <td>7.473077</td>
      <td>6.585065</td>
      <td>6.641667</td>
      <td>7.152632</td>
      <td>7.197826</td>
      <td>7.238889</td>
    </tr>
    <tr>
      <th>PlayStation 3</th>
      <td>6.853819</td>
      <td>7.306154</td>
      <td>6.820988</td>
      <td>7.710938</td>
      <td>7.735714</td>
      <td>7.350000</td>
      <td>7.436111</td>
      <td>6.978571</td>
      <td>7.219553</td>
      <td>7.142857</td>
      <td>7.485816</td>
      <td>7.355172</td>
    </tr>
    <tr>
      <th>PlayStation 4</th>
      <td>7.550000</td>
      <td>7.835294</td>
      <td>7.388571</td>
      <td>7.280000</td>
      <td>8.390909</td>
      <td>7.400000</td>
      <td>7.944000</td>
      <td>7.590000</td>
      <td>7.804444</td>
      <td>9.250000</td>
      <td>7.430000</td>
      <td>6.566667</td>
    </tr>
    <tr>
      <th>PlayStation Portable</th>
      <td>6.467797</td>
      <td>7.000000</td>
      <td>6.938095</td>
      <td>6.822222</td>
      <td>7.194737</td>
      <td>6.726667</td>
      <td>6.817778</td>
      <td>6.401961</td>
      <td>7.071053</td>
      <td>6.761538</td>
      <td>6.956790</td>
      <td>6.550000</td>
    </tr>
    <tr>
      <th>PlayStation Vita</th>
      <td>7.173077</td>
      <td>6.133333</td>
      <td>8.057143</td>
      <td>7.527273</td>
      <td>8.568750</td>
      <td>8.250000</td>
      <td>7.337500</td>
      <td>6.300000</td>
      <td>7.660000</td>
      <td>5.725000</td>
      <td>7.130000</td>
      <td>8.900000</td>
    </tr>
    <tr>
      <th>Wii</th>
      <td>6.262718</td>
      <td>7.294643</td>
      <td>6.234043</td>
      <td>6.733333</td>
      <td>7.054255</td>
      <td>6.426984</td>
      <td>7.410345</td>
      <td>5.011667</td>
      <td>6.479798</td>
      <td>6.327027</td>
      <td>5.966901</td>
      <td>6.975000</td>
    </tr>
    <tr>
      <th>Wireless</th>
      <td>7.041699</td>
      <td>7.312500</td>
      <td>6.972414</td>
      <td>6.740000</td>
      <td>7.509091</td>
      <td>7.360550</td>
      <td>8.260000</td>
      <td>6.898305</td>
      <td>6.906780</td>
      <td>7.802857</td>
      <td>7.417699</td>
      <td>7.542857</td>
    </tr>
    <tr>
      <th>Xbox</th>
      <td>6.819512</td>
      <td>7.479032</td>
      <td>6.821429</td>
      <td>7.029630</td>
      <td>7.303448</td>
      <td>5.125000</td>
      <td>8.277778</td>
      <td>7.021591</td>
      <td>7.485417</td>
      <td>7.155556</td>
      <td>7.884397</td>
      <td>7.313333</td>
    </tr>
    <tr>
      <th>Xbox 360</th>
      <td>6.719048</td>
      <td>7.137838</td>
      <td>6.857353</td>
      <td>7.552239</td>
      <td>7.559574</td>
      <td>7.141026</td>
      <td>7.650000</td>
      <td>6.996154</td>
      <td>7.338153</td>
      <td>7.325000</td>
      <td>7.317857</td>
      <td>7.112245</td>
    </tr>
    <tr>
      <th>Xbox One</th>
      <td>7.702857</td>
      <td>7.566667</td>
      <td>7.254545</td>
      <td>7.171429</td>
      <td>6.733333</td>
      <td>8.100000</td>
      <td>8.291667</td>
      <td>8.163636</td>
      <td>8.020000</td>
      <td>7.733333</td>
      <td>7.331818</td>
      <td>8.500000</td>
    </tr>
    <tr>
      <th>iPhone</th>
      <td>6.865445</td>
      <td>7.764286</td>
      <td>7.745833</td>
      <td>6.087500</td>
      <td>7.471930</td>
      <td>7.810784</td>
      <td>7.185185</td>
      <td>7.315789</td>
      <td>6.995588</td>
      <td>7.328571</td>
      <td>7.152174</td>
      <td>7.534921</td>
    </tr>
  </tbody>
</table>
</div>



The dataset that you've just printed shows the average score, by platform and genre.  Use the data to answer the questions below.


```python
print(ign_data.loc['PC',])
print(ign_data.loc['PC',].max())
```

    Action               6.805791
    Action, Adventure    7.334746
    Adventure            7.136798
    Fighting             7.166667
    Platformer           7.410938
    Puzzle               6.924706
    RPG                  7.759930
    Racing               7.032418
    Shooter              7.084878
    Simulation           7.104889
    Sports               6.902424
    Strategy             7.310207
    Name: PC, dtype: float64
    7.759930313588847



```python
print(ign_data.loc['PlayStation Vita',])
print(ign_data.loc['PlayStation Vita',].min())
```

    Action               7.173077
    Action, Adventure    6.133333
    Adventure            8.057143
    Fighting             7.527273
    Platformer           8.568750
    Puzzle               8.250000
    RPG                  7.337500
    Racing               6.300000
    Shooter              7.660000
    Simulation           5.725000
    Sports               7.130000
    Strategy             8.900000
    Name: PlayStation Vita, dtype: float64
    5.725



```python
# Fill in the line below: What is the highest average score received by PC games,
# for any genre?
high_score = 7.759930

# Fill in the line below: On the Playstation Vita platform, which genre has the 
# lowest average score? Please provide the name of the column, and put your answer 
# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)
worst_genre = 'Simulation'

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

## Step 3: Which platform is best?

Since you can remember, your favorite video game has been [**Mario Kart Wii**](https://www.ign.com/games/mario-kart-wii), a racing game released for the Wii platform in 2008.  And, IGN agrees with you that it is a great game -- their rating for this game is a whopping 8.9!  Inspired by the success of this game, you're considering creating your very own racing game for the Wii platform.

#### Part A

Create a bar chart that shows the average score for **racing** games, for each platform.  Your chart should have one bar for each platform. 


```python
# Bar chart showing average score for racing games by platform
# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Add title
plt.title("Average Score for Racing Games")

# Bar chart showing average score
sns.barplot(x=ign_data.index, y=ign_data['Racing'])

# Add label
plt.xlabel("Platform")
plt.xticks(rotation=45)
plt.ylabel("Score")

# Check your answer
step_3.a.check()

# plotting
plt.show()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



    
![png](/assets/images/exercise-bar-charts-and-heatmaps_files/exercise-bar-charts-and-heatmaps_16_2.png)
    



```python
# Lines below will give you a hint or solution code
#step_3.a.hint()
#step_3.a.solution_plot()
```

#### Part B

Based on the bar chart, do you expect a racing game for the **Wii** platform to receive a high rating?  If not, what gaming platform seems to be the best alternative?


```python
step_3.b.hint()
```


    <IPython.core.display.Javascript object>



<span style="color:#3366cc">Hint:</span> Check the length of the bar corresponding to the **Wii** platform.  Does it appear to be longer than the other bars?  If so, you should expect a Wii game to perform well!



```python
# Check your answer (Run this code cell to receive credit!)
step_3.b.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc99">Solution:</span> Based on the data, we should not expect a racing game for the Wii platform to receive a high rating.  In fact, on average, racing games for Wii score lower than any other platform.  Xbox One seems to be the best alternative, since it has the highest average ratings.


## Step 4: All possible combinations!

Eventually, you decide against creating a racing game for Wii, but you're still committed to creating your own video game!  Since your gaming interests are pretty broad (_... you generally love most video games_), you decide to use the IGN data to inform your new choice of genre and platform.

#### Part A

Use the data to create a heatmap of average score by genre and platform.  


```python
# Heatmap showing average game score by platform and genre
# Set the width and height of the figure
plt.figure(figsize=(12, 12))

# Add title
plt.title("Average Game Score by Platform and Genre")

# Heatmap showing average game score
sns.heatmap(data=ign_data, annot=True)

# Add label for horizotal axis
plt.xlabel("Genre")
plt.xticks(rotation=45)

# Check your answer
step_4.a.check()

# plotting
plt.show()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc33">Correct</span>



    
![png](/assets/images/exercise-bar-charts-and-heatmaps_files/exercise-bar-charts-and-heatmaps_22_2.png)
    



```python
# Lines below will give you a hint or solution code
#step_4.a.hint()
step_4.a.solution_plot()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc99">Solution:</span> 
```python
# Set the width and height of the figure
plt.figure(figsize=(10,10))
# Heatmap showing average game score by platform and genre
sns.heatmap(ign_data, annot=True)
# Add label for horizontal axis
plt.xlabel("Genre")
# Add label for vertical axis
plt.title("Average Game Score, by Platform and Genre")

```



    
![png](/assets/images/exercise-bar-charts-and-heatmaps_files/exercise-bar-charts-and-heatmaps_23_2.png)
    


#### Part B

Which combination of genre and platform receives the highest average ratings?  Which combination receives the lowest average rankings?


```python
step_4.b.hint()
```


    <IPython.core.display.Javascript object>



<span style="color:#3366cc">Hint:</span> To find the highest average ratings, look for the largest numbers (or lightest boxes) in the heatmap.  To find the lowest average ratings, find the smallest numbers (or darkest boxes).



```python
# Check your answer (Run this code cell to receive credit!)
step_4.b.solution()
```


    <IPython.core.display.Javascript object>



<span style="color:#33cc99">Solution:</span> **Simulation** games for **Playstation 4** receive the highest average ratings (9.2). **Shooting** and **Fighting** games for **Game Boy Color** receive the lowest average rankings (4.5).


# Keep going

Move on to learn all about **[scatter plots](https://www.kaggle.com/alexisbcook/scatter-plots)**!

---




*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/data-visualization/discussion) to chat with other learners.*
