---
classes: wide
title: "[GAN] Training Handwriting"

comments: true

---


Handwritten Digits - MNIST GAN With Seed Experiments
===
#### Make Your First GAN With PyTorch, 20202



```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
```

Dataset class
===


```python
class MnistDataset(Dataset):

  def __init__(self, csv_file):
    self.data_df = pd.read_csv(csv_file, header=None)
    pass

  def __len__(self):
    return len(self.data_df)

  def __getitem__(self, index):
    # 이미지 목표(레이블)
    label = self.data_df.iloc[index, 0]
    target = torch.zeros((10))
    target[label] = 1.0

    # 0~255의 이미지를 0~1로 정규화
    image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0

    # 레이블, 이미지 데이터 텐서, 목표 텐서 반환
    return label, image_values, target

  def plot_image(self, index):
    img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
    plt.title("label = " + str(self.data_df.iloc[index, 0]))
    plt.imshow(img, interpolation='none', cmap='Blues')
    pass

  pass

```


```python
# load data
mnist_dataset = MnistDataset("/content/drive/MyDrive/ColabNotebooks/myo_gan/mnist_data/mnist_train.csv")
```


```python
# check data contain images
mnist_dataset.plot_image(17)
```


![png](../assets/images/07_training_handwriting_files/07_training_handwriting_6_0.png)


Data Functions
===


```python
# functions to generate random data

def generate_random_image(size):
  random_data = torch.rand(size)
  return random_data

def generate_random_seed(size):
  random_data = torch.randn(size)
  return random_data
```

Discriminator Network
===


```python
from pandas.core.frame import DataFrame
# discriminator class

class Discriminator(nn.Module):

  def __init__(self):
    # initialise parent pytorch class
    super().__init__()

    # define neural network layers
    self.model = nn.Sequential(
        nn.Linear(784, 200),
        nn.Sigmoid(),
        nn.Linear(200, 1),
        nn.Sigmoid()
    )

    # create loss function
    self.loss_function = nn.BCELoss()

    # create optimiser, stochastic gradient descent
    self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

    # counter and accumulator for progress
    self.counter = 0
    self.progress = []

    pass

  def forward(self, inputs):
    # simply run model
    return self.model(inputs)

  def train(self, inputs, targets):
    # calculate the output of the network
    outputs = self.forward(inputs)

    # calculate loss
    loss = self.loss_function(outputs, targets)

    # increase counter and accumulate error every 10
    self.counter += 1
    if (self.counter % 10 == 0):
      self.progress.append(loss.item())
      pass
    if (self.counter % 10000 == 0):
      print("counter = ", self.counter)
      pass

    # zero gradients, perform a backward pass, update weights
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()

    pass

  def plot_progress(self):
    df = pd.DataFrame(self.progress, columns=['loss'])
    df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
    pass

  pass
```

Test Discriminator
===


```python
%%time
# test dicriminator can seperate real data from random noise

D = Discriminator()

for label, image_data_tensor, target_tensor in mnist_dataset:
  # real data
  D.train(image_data_tensor, torch.FloatTensor([1.0]))
  # fake data
  D.train(generate_random_image(784), torch.FloatTensor([0.0]))
  pass

```

    counter =  10000
    counter =  20000
    counter =  30000
    counter =  40000
    counter =  50000
    counter =  60000
    counter =  70000
    counter =  80000
    counter =  90000
    counter =  100000
    counter =  110000
    counter =  120000
    CPU times: user 1min 58s, sys: 1.29 s, total: 1min 59s
    Wall time: 2min 1s



```python
D.plot_progress()
```


![png](../assets/images/07_training_handwriting_files/07_training_handwriting_13_0.png)



```python
# manually run discriminator to check it can tell real data from fake

for i in range(4):
  image_data_tensor = mnist_dataset[random.randint(0, 60000)][1]
  print(D.forward(image_data_tensor).item())
  pass

for i in range(4):
  print(D.forward(generate_random_image(784)).item())
  pass
```

    0.9999792575836182
    0.9999396800994873
    0.99997878074646
    0.9999793767929077
    3.7969151890138164e-05
    4.6697099605808035e-05
    3.3929554774658754e-05
    2.329997187189292e-05



```python
class Generator(nn.Module):

  def __init__(self):
    # 파이토치 부모 클래스 초기화
    super().__init__()

    # 신경망 레이어 정의
    self.model = nn.Sequential(
        nn.Linear(1, 200),
        nn.Sigmoid(),
        nn.Linear(200, 784),
        nn.Sigmoid()
    )

    # SGD 옵티마이저 설정
    self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    # 진행 측정을 위한 변수 초기화
    self.counter = 0
    self.progress = []

    pass

  def forward(self, inputs):
    # 모델 실행
    return self.model(inputs)

  def train(self, D, inputs, targets):
    # 신경망 출력 계산
    g_output = self.forward(inputs)

    # 판별기로 전달
    d_output = D.forward(g_output)

    # 오차 계산
    loss = D.loss_function(d_output, targets)

    # 카운터를 증가시키고 10회마다 오차 저장
    self.counter += 1
    if (self.counter % 10 == 0):
      self.progress.append(loss.item())
      pass

    # 기울기를 초기화하고 역전파 후 가중치 갱신
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    pass

  def plot_progress(self):
    df = pd.DataFrame(self.progress, columns=['loss'])
    df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
    pass

  pass

```

Test Generator Output
===


```python
# check the generator output is of the right type and shape

G = Generator()

output = G.forward(generate_random_seed(1))
img = output.detach().numpy().reshape(28, 28)
plt.imshow(img, interpolation='none', cmap='Blues')
```




    <matplotlib.image.AxesImage at 0x7f769fbc3610>




![png](../assets/images/07_training_handwriting_files/07_training_handwriting_17_1.png)


Train GAN
===


```python
%%time 

# create Discriminator and Generator

D = Discriminator()
G = Generator()


# train Discriminator and Generator

for label, image_data_tensor, target_tensor in mnist_dataset:
  # train discriminator on true
  D.train(image_data_tensor, torch.FloatTensor([1.0]))
    
  # train discriminator on false
  # use detach() so gradients in G are not calculated
  D.train(G.forward(generate_random_seed(1)).detach(), torch.FloatTensor([0.0]))
    
  # train generator
  G.train(D, generate_random_seed(1), torch.FloatTensor([1.0]))

  pass

```

    counter =  10000
    counter =  20000
    counter =  30000
    counter =  40000
    counter =  50000
    counter =  60000
    counter =  70000
    counter =  80000
    counter =  90000
    counter =  100000
    counter =  110000
    counter =  120000
    CPU times: user 3min 16s, sys: 2.5 s, total: 3min 18s
    Wall time: 3min 24s



```python
# plot discriminator error

D.plot_progress()
```


![png](../assets/images/07_training_handwriting_files/07_training_handwriting_20_0.png)



```python
# plot generator error

G.plot_progress()
```


![png](../assets/images/07_training_handwriting_files/07_training_handwriting_21_0.png)



```python
# plot several outputs from the trained generator

# plot a 3 column, 2 row array of generated images
f, axarr = plt.subplots(2, 3, figsize=(16, 8))
for i in range(2):
  for j in range(3):
    output = G.forward(generate_random_seed(1))
    img = output.detach().numpy().reshape(28, 28)
    axarr[i, j].imshow(img, interpolation='none', cmap='Blues')
    pass
  pass
```


![png](../assets/images/07_training_handwriting_files/07_training_handwriting_22_0.png)



```python

```

{% if page.comments != false %}
{% include disqus.html %}
{% endif %}
