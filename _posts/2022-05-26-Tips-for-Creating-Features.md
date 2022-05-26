---
classes: wide

title: "Tips for Creating Features"

excerpt: "Tips from kaggle's instructor 'Ryan Holbrook'."

categories:
- data

tags:
- feature
- data

---

> **Tips on Discovering New Features**
> * Understand the features. Refer to your dataset's data documentation, if available.<br>
> 피처들을 이해한다. 가능한 데이터셋의 데이터 문서들을 참조한다.<br><br>
> * Research the problem domain to acquire **domain knowledge**. If your problem is predicting house prices, do some research on real-estate for instance. Wikipedia can be a good starting point, but books and journal articles will often have the best information.<br>
> 도메인 지식을 얻기위해 문제 도메인을 연구한다. 만약 문제가 주택가격을 예측하는 것이라면, 예들들어 부동산에 관한 연구를 한다. 위키페디아는 좋은 시작점이 될 수 있지만, 책과 잡지기사에 종종 가장 좋은 정보가 있을 수 있다.<br><br>
> * Study previous work. Solution write-ups from past Kaggle competitions are a great resource.<br>
> 작업전에 공부한다. 과거 캐글대회의 솔루션 기록은 훌륭한 자원이다.<br><br>
> * Use data visualization. Visualization can reveal pathologies in the distribution of a feature or complicated relationships that could be simplified. Be sure to visualize your dataset as you work through the feature engineering process.<br>
> 데이터 시각화를 이용한다. 시각화는 단순화 될 수 있는 기능의 분포 또는 복작한 관계의 병리를 드러낼 수 있다. 피처 엔지니어링을 진행하면서 데이터셋을 시각화한다.<br><br>


> **Tips on Creating Features** <br>
> It's good to keep in mind your model's own strengths and weaknesses when creating features. Here are some guidelines:<br>
> * Linear models learn sums and differences naturally, but can't learn anything more complex.<br>
> 선형 모델은 자연스럽게 합과 차이는 학습하지만, 복잡한 것을 학습할 수 없다.<br><br>
> * Ratios seem to be difficult for most models to learn. Ratio combinations often lead to some easy performance gains.<br>
> 비율은 대부분 모델에 학습하기 어려워 보인다. 비율 복합은 종종 쉽게 좋은 성능을 가져온다.<br><br>
> * Linear models and neural nets generally do better with normalized features. Neural nets especially need features scaled to values not too far from 0. Tree-based models (like random forests and XGBoost) can sometimes benefit from normalization, but usually much less so.<br>
> 선형 모델과 신경망은 일반적으로 정규환 된것으로 더 잘 된다. 신경망은 특별히 0에서 멀지 않은 값으로 된 것이 필요하다. 트리기반 모델(랜덤포레스트와 XGBoost)은 종종 정규화에서 이득을 보지만, 일반적으로는 아주 적다.<br><br>
> * Tree models can learn to approximate almost any combination of features, but when a combination is especially important they can still benefit from having it explicitly created, especially when data is limited.<br>
> 트리 모델은 거의 모든 기능 조합을 근사화하는 방법을 학습 할 수 있지만 특별히 데이터가 제한적일 때 명시적으로 생성되는 이점을 누릴 수 있다.<br><br> 
> * Counts are especially helpful for tree models, since these models don't have a natural way of aggregating information across many features at once.<br>
> 카운트는 트리 모델에 특히 유용하다. 이 모델에는 한번에 많은 기능에 걸쳐 정보를 집계하는 자연스러운 방법이 없기 때문이다.<br><br>


This content is from ['Ryan Holbrook'](https://www.kaggle.com/code/ryanholbrook/creating-features) of Kaggle's instructor.