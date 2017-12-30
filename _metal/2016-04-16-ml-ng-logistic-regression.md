---
layout: post
title:  "Logical Regression"
date:   2016-04-15 0:30:00
categories: Machine Learning
description: Logistic Regression
published: false
---

# Logistic Regression

## Hypothesis Function

Logistic Regression에서 사용할 Hypothesis 함수는 다음과 같다.

$$ h _\theta (x) = g(\theta _T x) $$

여기서 $g$ 함수는 sigmoid 함수라 불리며, 다음과 같다.

$$ g(z) = \frac {1}{1 + e^{-z}}$$

<img class="col two center" src="/images/201604/600px-Logistic-curve.svg.png"/>

수평축 데이터에인 z 값에 따라서 출력이 1/0인 함수이다.

한편 여기서 $\theta$와 $x$는 벡터값이다. 즉, $z = \theta ^T x$이므로, 입력하는 feature들의 값이 학습된 $\theta$ 값을 통해 분류를 할 수 있게 된다.

## Cost Function

logistic Regression은 True/False로 특정데이터를 구분짓는 선형분석이다. Linear Regreesion의 경우 특정데이터의 분포를 바탕으로 데이터의 추세를 예측하기 위한 분석이라면, logistic regression의 경우 주어진 데이터가 원하는 출력에 대하여 True/False를 판별할 수 있도록 하는데 목적이 있다.

$$ J(\theta) = \frac {1}{m} \sum _{i = 1} ^m Cost(h _\theta(x^{(i)}), y^{(i)})$$

여기서,

$$ Cost(h _\theta(x^{(i)}), y^{(i)}) = \begin{cases}
-log(h _\theta(x)) & y = 1 \\
-log(1 - h _\theta(x)) & y = 0
\end{cases} $$

이므로, Cost function은 다음과 같이 구성된다.

$$ J(\theta) = \frac {1}{m} [ \sum _{i = 1} ^m -log(h _\theta(x^{(i)}) + (1-y^{(i)}) log (1 - h _\theta (x ^{(i)})) $$

## Gradient Descent

Linear Regression과 동일하게, Cost function의 최소지점을 gradient를 이용하여 찾는다

즉, 다음의 절차를 모든 $\theta _j$에 대하여 업데이트를 하는 것이다.

$$ \theta _j := \theta _j - \alpha \frac {\delta}{\delta \theta _j} J(\theta) $$


사실 Gradient Descent와 같은 최적 위치를 찾는 방법은 Gradient Descent외에도 다양한 방법이 있으나 지금 이 시점에서는 다루지 않는다.

## Multiclass Classification

기본적인 Logistic Regression에 의해서는 데이터를 True/False를 구별할 수 있는 하나의 기준만을 제공하는데, 데이터는 두 개 이상의 분류를 할 수 있는 경우가 많이 있다.

<img class="col two center" src="/images/201604/11AB5396-A696-4A1D-9233-D133DDFCB07E.png"/>
<div class="caption">
단, 두 종류의 분류만을 한다면 **Binary Classification**이라고 부르며, 그 이상의 분류를 하는 경우 **Multiclass Classification**이라고 한다.
</div>

이러한 경우에 적절한 Classification 전략은 **One-vs-all** 또는 (one-vs-rest)이다. 즉, 관심있는 한가지를 분류하고 나머지는 부정한 뒤에, 다시 다른 관심있는 데이터에 대해서 분류를 하는 것이다.

즉, 위 그림의 Multi-class Classification의 경우 아래 그림과 같이 세차례에 걸쳐서 각기 다른 hypothesis function을 수립하여 분류를 진행한다.

<img class="col two center" src="/images/201604/235FB5D3-6A6F-4B08-A348-D0C15BE7E3B1.png"/>
