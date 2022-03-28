<a href="https://colab.research.google.com/github/mucollabo/firstGAN/blob/main/06_gan_simple_pattern.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import random
```


```python
def generate_real():
  real_data = torch.FloatTensor(
      [random.uniform(0.8, 1.0),
       random.uniform(0.0, 0.2),
       random.uniform(0.8, 1.0),
       random.uniform(0.0, 0.2)])
  return real_data

generate_real()
```




    tensor([0.9801, 0.0542, 0.9029, 0.1930])




```python
class Discriminator(nn.Module):

  def __init__(self):
    # 파이토치 부모 클래스 초기화
    super().__init__()
    # 신경망 레이어 정의
    self.model = nn.Sequential(
        nn.Linear(4, 3),
        nn.Sigmoid(),
        nn.Linear(3, 1),
        nn.Sigmoid()
    )

    # 손실함수 설정
    self.loss_function = nn.MSELoss()

    # SGD 옵티마이저 설정
    self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

    # 진행 측정을 휘한 변수 초기화
    self.counter = 0;
    self.progress = []

    pass

  def forward(self, inputs):
    # 모델 실행
    return self.model(inputs)

  def train(self, inputs, targets):
    # 신경망 출력 계산
    outputs = self.forward(inputs)

    # 손실 계산
    loss = self.loss_function(outputs, targets)

    # 카운터를 증가시키고 10회마다 오차 저장
    self.counter += 1
    if (self.counter % 10 == 0):
      self.progress.append(loss.item())
      pass
    if (self.counter % 10000 == 0):
      print("counter = ", self.counter)
      pass

    # 기울기를 초기화하고 역전파 후 가중치 갱신
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    pass

  def plot_progress(self):
     df = pd.DataFrame(self.progress, columns=['loss'])
     df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
     pass
  pass


```


```python
def generate_random(size):
  random_data = torch.rand(size)
  return random_data
  
```


```python
D = Discriminator()

for i in range(10000):
  # 실제 데이터
  D.train(generate_real(), torch.FloatTensor([1.0]))
  # 생성된 데이터
  D.train(generate_random(4), torch.FloatTensor([0.0]))

  pass
```

    counter =  10000
    counter =  20000



```python
D.plot_progress()
```


![png](/Users/charles/Documents/GitHub/mucollabo.github.io/assets/images/06_gan_simple_pattern_files/06_gan_simple_pattern_6_0.png)



```python
# manually run discriminator to check it can tell real data from fake
print(D.forward(generate_real()).item())
print(D.forward(generate_random(4)).item())
```

    0.7629599571228027
    0.06228242069482803



```python
class Generator(nn.Module):

  def __init__(self):
    super().__init__()

    # 신경망 레이어 정의
    self.model = nn.Sequential(
        nn.Linear(1, 3),
        nn.Sigmoid(),
        nn.Linear(3, 4),
        nn.Sigmoid()
    )

    # SGD 옵티마이저 설정
    self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    # 진행 측정을 위한 변수 초기화
    self.counter = 0;
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
    self.counter += 1;
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
    df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
    pass

  pass

```


```python
G = Generator()
G.forward(torch.FloatTensor([0.5]))

```




    tensor([0.5871, 0.6674, 0.5653, 0.3294], grad_fn=<SigmoidBackward0>)




```python
%%time

# 새로운 판별기 및 생성기 생성

D = Discriminator()
G = Generator()

image_list = []

# 판별기와 생성기 훈련

for i in range(10000):

  # 1단계: 참에 대해 판별기 훈련
  D.train(generate_real(), torch.FloatTensor([1.0]))

  # 2단계: 거짓에 대해 판별기 훈련
  # G의 기울기가 계산되지 않도록 detach() 함수를 이용
  D.train(G.forward(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))

  # 3단계: 생성기 훈련
  G.train(D, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))

  # 매 1000회 이미지를 저장
  if (i % 1000 == 0):
    image_list.append(G.forward(torch.FloatTensor([0.5])).detach().numpy())


  pass


```

    counter =  10000
    counter =  20000
    CPU times: user 12.3 s, sys: 10.3 ms, total: 12.3 s
    Wall time: 12.3 s



```python
D.plot_progress()
```


![png](/Users/charles/Documents/GitHub/mucollabo.github.io/assets/images/06_gan_simple_pattern_files/06_gan_simple_pattern_11_0.png)



```python
G.plot_progress()
```


![png](/Users/charles/Documents/GitHub/mucollabo.github.io/assets/images/06_gan_simple_pattern_files/06_gan_simple_pattern_12_0.png)



```python
G.forward(torch.FloatTensor([0.5]))
```




    tensor([0.9436, 0.0366, 0.9376, 0.0356], grad_fn=<SigmoidBackward0>)




```python
import numpy as np

plt.figure(figsize = (16, 8))
plt.imshow(np.array(image_list).T, interpolation='none', cmap='Blues')
```




    <matplotlib.image.AxesImage at 0x7ff5bae7d090>




![png](/Users/charles/Documents/GitHub/mucollabo.github.io/assets/images/06_gan_simple_pattern_files/06_gan_simple_pattern_14_1.png)



```python

```
