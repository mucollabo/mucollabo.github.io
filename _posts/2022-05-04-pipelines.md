---
classes: wide

title: "Pipelines"

excerpt: "Pipelines are a simple way to keep your data preprocessing and modeling code organized."

categories:
- AI

tags:
- data

---
![png](https://global-uploads.webflow.com/60ddb7e2e50eaef5bec9595c/6167204ed759bf9693f7ee1b_data-pipelines2.png)
<p align="center">Photo by <a href="https://www.acceldata.io/blog/why-manage-data-pipelines-not-data-warehouses">acceldata</a></p>
>파이프라인은 데이터 전처리와 모델링 코드를 만드는데 심플한 방식이다. 특별히, 파이프라인은 전처리와 모델링 단계를 번들로 제공하여 번들이 하나의 단계처럼 사용할 수 있다.
>많은 데이터 사이언스들이 파이프라인 없이 모델링을 마구하지만, 파이프라인은 중요한 장점들이 있다.
>> 1.**Clean Code**: 데이터 전처리의 각 단계를 설명하는 것은 지저분해질 수 있다. 파이프라인을 사용하면 각 단계의 training 과 validation 데이터를 수동으로 추적할 필요가 없다.
>>
>> 2.**Fewer Bugs**: 단계를 빼먹거나 또는 전처리 단계를 잊어버릴 기회가 줄어든다.
>>
>> 3.**Easier to Productionize**: 프로토타입에서 대규모 개발단계로 전화하는 것은 매우 어려울 수 있다. 하지만 파이프라인은 이것을 도와줄 수 있다.
>>
>> 4.**More Options for Model Validation**: 교차검증을 다루는 옵션들이 있다.
> 

출처: [kaggle](https://www.kaggle.com/code/alexisbcook/pipelines)