---
layout: post
title:  "Linear Regression - I"
date:   2016-03-30 2:00:00
categories: Machine Learning
description: Linear Regression for single & multiple variables
published: false
tags:
- Machine Learning
- Linear Regression
- Coursera
---

# Linear Regression

## 변수 하나(one variable)에 대한 Linear Regression

### 모델
앞서 설명한대로 **Regression** 문제는 연속적인 결화를 도출하는 함수에 대한 예측을 하는 문제이다. 여기서 살펴볼 문제는 변수 하나에 대한 Regression이므로 Univariate linear regression이라고도 불린다.

단일 변수에 대한 Linear Regression은 하나의 입력값에 대한 하나의 출력을 특징인 예측을 하는 경우에 유용하다. 또한 Supervised Learning을 사용하는데, 이는 우리가 특정 입력에 대하여 어떤 결과가 나와야 하는지 알고 있기 때문이다.

### Hypothesis function
입력에 대한 출력은 다음과 같이 나타낼 수 있다.
$$ h_\theta (x) = \theta_1 + \theta_1 x $$

주어진 $ h_\theta (x)$에 대하여 $\theta_0$과 $\theta_1$를 주어졌을때 우리는 출력값인 $y$를 알 수 있을 것이다. 즉, 우리는 여기서 $h_\theta (x)$ 함수를 만들 것인데, 그것은 우리가 가진 입력 데이터(x)를 우리가 가진 출력 데이터(y) 간의 상관관계를 표현해 줄 것이다.

여기서 $x$는 입력 데이터 변수이며, $\theta$는 feature이다.

예를 들어, 다음과 같은 데이터가 주어졌다고 하자.

| x:input | y:output |
|:---:|:---:|
| 0 | 4 |
| 1 | 7 |
| 2 | 7 |
| 3 | 8 |

여기서 임의로 $h_\theta(x)$ 함수에서 $h_0 = 2$이고 $h_1 = 2$라고 가정(hypothesis)해보자. 이런 경우에 대하여 hypothesis function은 다음과 같다.

$$h_\theta(x) = 2 + 2x$$

그러면 입력 값 1에 대하여, $h_\theta(1)=4$가 되는데 이는 우리가 알고 있는 값보다 3이 적은 값이다. 이렇게 hypothesis function에 의해서 예측한 결과는 오차가 생기게 되는데, 이를 표현하는 방법을 Cost Function이라고 한다.

### Cost Function
hypothesis function의 정밀함은 *cost function*을 통해 알 수 있다. 이것은 입력에 대한 hypotheis function의 결과와 실제 결과 값 간의 오차의 합의 평균(MSE: [Mean Square Error](https://en.wikipedia.org/wiki/Mean_squared_error))이다. 간단히 분산이라고 생각해도 되는데 약간 차이는 있다.

Cost function은 다음과 같이 표현할 수 있다.

$$J(\theta _0, \theta _1) = \frac {1} {2m} \sum^m_{i=1} ((h_\theta (x)) - y(i))^2$$
여기서 m은 데이터의 갯수이다.

이 식은 기본적인 MSE 공식에서 $n$이 아니라 $2m$을 사용하였는데, 이는 나중에 사용할 gradient descedent 연산서 간단히 제거가 되기 때문이다.

이제 여기서 우리가 원래 찾으려고 하는 답은 `최적의 hypothesis function`은 결국 `오차가 가장 작은 함수`를 의미한다는 것을 알 수 있따. 따라서 $J(\theta _0, \theta _1)$를 최소화 한 hypothesis function이 우리가 찾아야 하는 답이다.

### Gradient Descent
이제 어떻게 하면 hypothesis function의 정확도를 스스로 높일 수 있는지 알아보자. 앞서 우리는 hypothesis function과 그것의 정확도를 평가할 수 있는 cost function에 대해서 알아보았다. 그리고 함수의 최저값을 구하는데는 미분을 사용할 것이고, 여기서 *gradient*를 사용할 것이다.

이제 우리는 hypothesis function (= $h_\theta (x) = \theta _0 + \theta _1 (x)$)에서 $\theta _0$과 $\theta _1$에 집중할 것이다. 그리고 그 둘의 상관관계에 논할 것이다. $\theta _0$를 $x$축에 $\theta _1$를 $y$축에 두고, cost function의 값을 $z$ 축에 두었을때 3차원의 굴곡진 평면을 생각할 수 있다. 그러면 그 평면에서 가장 아래에 있는 점의 위치가 우리가 찾고자 하는 **minimized cost function**의 위치일 것이다.

이 최저점을 찾는 것은 결국 cost function에 대하여 `편미분`을 하는 것이며, cost function의 평면 상 임의의 지점에서 시작해서 미분을 통해 얻은 기울기 벡터값에 따라 이동해가면서 최저점의 위치를 찾는다. 이때 움직이는 길이의 크기를 `learning rate`라고 하고 $\alpha$로 나타낸다.

Gradient descent 식은 다음과 같다.

$$ \theta _j = \theta _j + \alpha \frac {\delta} {\delta \theta _j} J(\theta _0, \theta _1)$$ for $j = 0, j = 1$

위에서 계산하는 값이 수렴하도록 해야한다.

### Gradient Descent의 Linear Regression 적용
결론적으로 cost function의 gradient를 갖고 어떻게 최적의 hypothesis function을 얻기 위해선 오차를 더 적게 gradient에 의한 결과를 cost function에 반영해야 한다.

따라서 수렴할때까지 다음의 과정을 계속한다.

$$ \theta _0 = \theta _0 - \alpha \frac {1} {m} \sum ^m _{i=1} (h _\theta (x(i)) - y(i)) x _0$$

$$ \theta _1 = \theta _1 - \alpha \frac {1} {m} \sum ^m _{i=1} (h _\theta (x(i)) - y(i)) x _1$$

여기서 m은 학습할 데이터의 수이며, $\theta _0$와 $\theta _1$은 hypothesis function의 상수와 파라미터이며, $x(i)$, $y(i)$는 주어진 학습 데이터이다.

결론적으로 위 과정(gradient descent)을 반복할수록 우리의 hypotheis function은 오차가 적어지고 정확해져갈 것이다.

## 여러개의 변수(Multiple Variable)에 대한 Linear Regression

### 모델
앞서 설명한 Linear Regression은 단일 변수(x)에 대한 regression으로서 1차 hypothesis function을 찾는 것에 대해 알아보았다. 그렇다면 여러개의 변수가 주어질 때에는 어떻게 하는 것인지 알아보기로 하자.

예를 들어, 집의 가격과 집의 크기 간의 관계를 살펴보았던 과거의 경우를 살펴보자.

| X | Y |
|:---:|:---:|
| 집의 크기 | 집의 가격 |

하지만 집의 가격에는 집의 크기 뿐만 아니라 다른 것들도 영향을 미칠 수 있다.

<table>
  <thead>
    <tr>
      <th style="text-align: center" colspan="2">X</th>
      <th style="text-align: center;">Y</th>
    </tr>
  </thead>
  <tr>
    <td style="text-align: center;">$x _1$</td>
    <td style="text-align: center;">집의 크기</td>
    <td style="text-align: center;" rowspan="4">집의 가격</td>
  </tr>
  <tr>
    <td style="text-align: center;">$x _2$</td>
    <td style="text-align: center;">침실 개수</td>

  </tr>
  <tr>
    <td style="text-align: center;">$x _3$</td>
    <td style="text-align: center;">층수</td>

  </tr>
  <tr>
    <td style="text-align: center;">$x _4$</td>
    <td style="text-align: center;">건축 연도</td>
  </tr>
</table>

이렇게 여러개의 특징들이 결과 값에 영향을 미치는 경우를 **Multi-variable Linear Regression**이라고 한다.
그러면 앞에서 살펴본 hypothesis function, cost function, gradient descendent 는 어떻게 변하는지 살펴보자.

### Hypothesis function

#### Single Variable
$ h_\theta (x) = \theta_1 x_0 + \theta_1 x $

#### Multi-Variable $(n=4)$
$ h_\theta (x) = \theta_1 x_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \theta_4 x_4 $

통상적으로 $x_0$은 표기에서 생략되는데 $x_0 = 1$이다. $x_0$가 있음으로 인해 feature vector ($[\theta _0, \theta _1, \theta _2, ...])의 index가 0에서 시작하는 것처럼 입력 변수의 index도 0부터 시작할 수 있다. 이는 프로그래밍을 하는데 있어서 보다 generic한 코딩을 할 수 있게 해주어 코드를 간단하게 만들 수 있게 해준다.

### Cost Function

#### Single Variable

$J(\theta _0, \theta _1) = \frac {1} {2m} \sum ^m _{i=1} ((h _\theta (x)) - y(i))^2$

#### Multi-Variable $(n=4)$

$J(\theta) = \frac {1} {2m} \sum^m_{i=1} ((h_\theta (x^{(i)})) - y^{(i)})^2 \enspace where \enspace J(\theta) = J(\theta _0, \theta _1, ..., \theta _n)$

얼핏 보기에 식에는 큰 차이가 없어 보이지만, hypothesis function이 바뀌었다는 점을 염두에 두자.

### Gradient Descent

#### Single Variable

$\theta _j = \theta _j - \alpha \frac {1} {m} \sum ^m _{i=1} (h _\theta (x(i)) - y(i)) x _j(i)$

$simultaneously \enspace update \enspace \theta _0, \theta _1 \enspace for \enspace j = 0, j = 1$

#### Multi-Variable $(n=4)$

$\theta _j = \theta _j - \alpha \frac {1} {m} \sum ^m _{i=1} (h _\theta (x(i)) - y(i)) x _j(i)$

$simultaneously \enspace update \enspace \theta \enspace for \enspace j = 0, 1, ..., n$

여기서도 hypothesis가 변경되어서, 편미분을 수행해야하는 대상이 늘어났다는 점을 염두에 두어야 한다. 앞에서 다루었지만 여기서 $\frac {1} {m} \sum ^m _{i=1} (h _\theta (x(i)) - y(i)) x _j(i) = \frac {\delta} {\delta \theta _j} J(\theta)$ 임을 기억하자.

지금까지 Linear Regression의 수학적인 바탕을 살펴보았다. 편미분을 배웠다면 쉽게 상상할 수 있는 개념이다. 다음에는 Linear Regression을 하는데 있어서 고려해야 할 사항들을 다룰 것이다.
