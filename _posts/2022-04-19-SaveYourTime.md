---
title: "Save Your Time"

excerpt: "introduce useful Pandas Snippets"

categories:
  - data

tags:
  - python
  - pandas
  - snippet

last_modified_at: 2022-04-14T17:06:00-05:00
---

당신의 시간을 절약해주는 유용한 판다스 snippets을 소개한다.

* Snippet 1: pandas를 임포트 할 때 import pandas **<u>as pd</u>**
* Snippet 2: dataframe 생성

```python
  import pandas as pd
  
  df = pd.DataFrame({
                    ‘Name’: [‘Brian’, ‘Sarah’, ‘John’],
                    ‘Age’: [21, 25, 33],
                    ‘City’: [‘New York’, ‘Los Angeles’, ‘Chicago’]
                    })
```

* Snippet 3: dataframe의 데이터에 접근

```python
df['Age']
```

* Snippet 4: dataframe에 column 추가

```python
df['Occupation'] = ['Engineer', 'Teacher', 'Programmer']
```

* Snippet 5: dataframe에서 column 삭제

```python
del df['Age']
```

* Snippet 6: dataframe에 새로운 이름 column 생성

```python
df['Name'] = df['FirstName'] + '' + df['LastName']
```

* Snippet 7: dataframe에 row 추가

```python
df.append({
          'FirstName':'Alex',
          'Age':34,
          'City':'San Francisco'
          })
```

* Snippet 8: dataframe에서 row 삭제

```python
df.pop(2)
```

* Snippet 9: dataframe 정렬

```python
df.sort(column='Age')
```

* Snippet 10: dataframe 순서 반전

```python
df.reverse()
```

* Snippet 11: dataframe의 사이즈 찾기

```python
len(df)
```

* Snippet 12: 인덱스로 데이터에 접근

```python
df[2]
```

* Snippet 13: dataframe에 데이터 입력

```python
df.insert(0, {
              ‘FirstName’: ‘Joe’,
              ‘Age’: 45,
              ‘City’: ‘Phoenix’
              })
```

* Snippet 14: dataframe에서 데이터 제거

```python
df.del(['FirstName', 'City'])
```

* Snippet 15: dataframe을 순회하기

```python
for row in df:
    print(row)
```

* Snippet 16: dataframe에서 최대 값 찾기

```python
max(df['Age'])
```

* Snippet 17: dataframe에서 최소 값 찾기

```python
min(df['Age'])
```

![png](https://images.unsplash.com/photo-1511466422904-ecdbc68407a1?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1287&q=80)
<p>Photo by <a href="https://unsplash.com/@nixcreative"><u>Tyler Nix</u></a> on <a href="https://unsplash.com"><u>Unsplash</u></a></p> 

* Snippet 18: dataframe에서 평균값 찾기

```python
mean(df['Age'])
```

* Snippet 19: dataframe에서 표준편차 찾기

```python
std(df['Age'])
```

* Snippet 20: dataframe에서 중앙값 찾기

```python
median(df['Age'])
```

* Snippet 21: dataframe에서 분위수 찾기

```python
quantile(df['Age'], 0.5)
```

* Snippet 22: dataframe에서 데이터 그룹핑

```python
df.groupby('City')
```

* Snippet 23: dataframe에서 row 갯수 세기

```python
len(df)
```
<br>


this content is from -Alain Saamego- [meduim.com](https://medium.com/@alains/23-python-pandas-snippets-that-will-saveyou-time-368436894efe)

if you want to find that more detail, visit above link.

