---
layout: post
title:  "Linear Regression - II"
date:   2016-04-06 01:00:00
categories: Machine Learning
description: Linear Regression을 더 잘하기 위해서 고려해야 할 사항
published: false
tags:
- Machine Learning
- Linear Regression
- Coursera
---

강의에서 다뤄진 Linear Regression을 하는데 있어서 고려해야하는 내용은 다음과 같다.

1. <a href='feature scaling'>Feature Scaling</a>
2. <a href='learning rate'>Learning Rate</a>
3. <a href='f p regression'>Features and Polynomial Regression</a>
4. <a href='normal equation'>Normal Equation</a>

<a class='anckor' id='feature scaling'></a>

# Feature Scaling

여러개의 Feature가 있을 때, 유사한 Scale을 갖도록 Feature Normalization을 해주어야 한다.
만약 $\theta _1$이 $\theta _2$에 비해서 매우 크다면 아래 그림과 같이 일그러진 형태의 Cost 평면을 형성하게 된다. 이는 수렴하는 방향에도 영향을 주어, global minimum 값을 찾는 시간을 지연시키게 된다.
<img class="col one center" src="/images/201604/2F86C27F-0847-49DA-961E-64CECA157187.png"/>

그렇다면 적절한 Normalization 값은 어떤 값이어야 하는가.
Mean Normalization을 통해 Feature Scaling을 하는 것을 권한다. Mean Normalization을 했을때 장점은, Scaling 후의 Feautre의 평균값이 0이 되기 때문이다.

Mean Normalization이라고 하더라도 다음과 같이 몇가지 방법이 있을 수 있다.

1. 최대값 이용: $ X _i \leftarrow \frac {X _i - \mu} {X _max}
1. 범위 이용: $ X _i \leftarrow \frac {X _i - \mu _i} {X _max - X _min}$
1. 분산 이용: $ X _i \leftarrow \frac {X _i - \mu _i} {s _i}$


<a class='anckor' id='learning rate'></a>

# Learning Rate $\alpha$

Gradient Descent 식에서 보면,
$$ \theta _j = \theta _j + \alpha \frac {\delta} {\delta \theta _j} J(\theta _0, \theta _1)$$ for $j = 0, j = 1$$
$\alpha$ 값은 **learning rate**로서 gradient 값을 반영하는 인자로서, 값의 크기에 따라 Cost Function의 Global minimum을 찾는데 영향을 미치게 된다.

1. $\alpha$가 너무 큰 경우
  - global minimum을 찾지 못하게 된다.
1. $\alpha$가 너무 작은 경우
  - global minimum을 향해 가겠지만, iteration이 증가하여 learning time을 증가시킨다.
1. $\alpha$가 적당한 경우
  - 적절한 시점에 learning을 종료할 수 있다.

따라서, $\alpha$의 후보를 두고, 값을 바꿔가면서 수렴하는 정도를 살펴서 적절한 크기의 $\alpha$ 값을 선택해야 한다. 적당한 $\alpha$의 예로, 대략 3배씩 증가하는 값들을 예로 들 수 있다.
* 0.001, 0.003, 0.01, 0.03, 0.1, 0.3


<a class='anckor' id='f p regression'></a>

# Features and Polynomial Regression

## Features
learning에 유의미한 feature들을 선택한다. 또한 필요에 따라서 feature를 추가할수도 있다.

## Polynomial Regression

polynomial regresion을 사용하게 되면 선을 이용하는 것보다 더 유연하게 데이터에 더욱 가깝게 curve fitting을 할 수 있다. 데이터의 분포에 따라서 feature의 차수에 대한 힌트를 얻을 수 있다. 아래 도표와 같이 집의 크기와 가격에 대하여 빨간x와 같이 데이터가 분포해 있다고 해보자.

<img class='col two center' src="/images/201604/A06414E3-6BE0-44E3-A0C4-215C294B3477.png"/>

위 데이터에 대해서 일직선 형태의 linear regression을 적용하기 보다는 곡선 형태의 선으로 regression을 하는 것이 보다 정확한 분석이 될 것이다. 다만 size에 대한 해석을 하게 되므로 이 경우에는 1차 함수를 이용하기 보다는 size(feature)에 대한 polynoial equation을 수립하여 linear regression을 수행한다.

주어진 데이터에 잘 맞는 곡선은 **파란 선**이겠지만, 좀더 큰 크기에 대한 데이터가 더 주어진다면 곡선이 어떻게 변화할 것인지 추가로 정보를 얻을 수 있다. 저렴해질 수도(점선; 그럴리 없겠지만...),  가격이 더 커질수도(녹색) 있다. feature의 차수르 바꾸어 적용한다면 더 정확한 학습이 가능하다.

<a class='anckor' href="normal equation"></a>

# Normal Equation

Learning parameter $\theta$를 구하기 위해서 대표적으로 사용되는 것이 Gradient Descent이다. Coursera 강좌에서는 이것 되에도 직접적으로 $\theta$를 계산할 수 있는 방법으로 **Normal Equation**을 제시하고 있다.

이를테면, Gradient Descent는 Cost function의 gradient를 계속해서 추적해가면서 cost가 최소인 parameter를 찾는 방법이지만, Matrix 연산만으로도 $\theta$를 찾을 수 있는 것이다.

$$\theta = (X^T X)^{-1} X^T y$$

이 방법이 Gradient Descent와 비교하여 갖는 장단점은 다음과 같은 것들이 있다.

| Gradient Descent | Normal Equation |
|---|---|
|learning rate를 정해야 한다. | learning rate를 선택하지 않아도 된다. |
|iteration을 여러번 수행해야 한다. | iteration이 필요하지 않으므로, 수렴여부를 확인할 필요가 없다. |
|데이터가 많아도 잘 동작한다. | 데이터가 많을 경우 매우 느리다 |
| | Matrix Inversion을 해야한다. 특히 $(X^T X)^{-1}$연산은 $O(n^3)$이므로 연산량이 많은 방법이다 |

특히 샘플의 양이 많고 적음의 기준에 대해서, 1000 정도는 작은 데이터 수이고 10000개 정도면 Normal Equation을 적용하기엔 큰 데이터라고 한다.

또한 여기서 주의해야할 사항은 X가 Inverse가 가능해야 한다는 점이다. pseudo inverse를 이용해서 대부분 inverse를 할 수 있지만, 그래도 inverse가 가능하지 않은 singular matrix라면 다음 두가지 경우에 해당한다.

* feature 중복
  * 내용이 같은 feature가 포함되어 있을 수 있다.
* 너무 많은 featuer
  * feature에 비하여 데이터의 수가 적을 수 있다.
  * feature를 줄이거나 regularization을 한다.
