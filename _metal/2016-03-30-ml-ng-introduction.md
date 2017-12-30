---
layout: post
title:  "Machine Learning Introduction"
date:   2016-03-30 1:00:00
categories: Machine Learning
description: Machine Learning 소개
published: false
tags:
- Machine Learning
- Coursera
---

Coursera의 대표적인 강좌중 하나인 `Introduction to Machine Learning`에 대한 정리를 하고자 한다. 지난 2월 3일 Certification을 받았고, 지금은 Deep Learning을 공부하는 중이지만, 다시한번 정리할 필요가 있음을 절실하게 느끼고 있었다. 다행히도 강의를 들으면서 틈틈이 정리해놓은 노트가 있었기에 아마도 그것을 보면서 이해가 가지 않았던 것, 또는 설명이 필요한 것들 등에 살을 붙여가면서 정리를 하게 되지 않을까 싶다.

# Introduction

## 머신러닝(Machine Learning)이란 무엇인가
* 컴퓨터로 하여금 명시적인 프로그램 없이 학습할 수 있는 능력을 갖도록 하는 분야 (old)
* 경험(E)으로부터 배울 수 있는 프로그램으로서, 경험을 위해 수행하는 작업(T)과 그것의 결과(P)를 통해 경험의 질이 향상되는 프로그램

예) 체커 게임
 * E = 체커 게임을 여러번 해본 경험
 * T = 체커 게임 1회
 * P = 프로그램이 다음 게임에서 승리할 확률

결국은 원하는 결과를 얻는데 필요한 과정에 대하여 명확한 프로그래밍 없이, Software가 스스로 원하는 결과를 얻을 수 있도록 해주는 절차를 머신러닝이라고 한다.

## Supervised Learning (지도학습)
Supoervised Learning은 주어진 데이터에서 우리가 어떤 것이 원하는 결과인지 알고 있는 상황에서 입력과 출력간의 관계를 찾는 것을 말한다.

Supervised Learning 문제는 다음과 같이 분류된다.

* Regression
  * 연속적인 데이터 결과를 예측하는 문제. 입력으로부터 연속적인 함수를 도출해 낸다.

* Classification
  * 이산적인 데이터 결과를 예측하는 문제. 입력으로부터 정해진 수 만큼의 분류를 한다.

예1) 부동산 시장에서 집의 크기에 대한 데이터가 주어졌을때, 집의 가격을 예측하는 문제. 집의 가격과 크기 간의 상관관계를 나타내는 연속적인 데이터를 낼 수 있는 함수를 결과로 얻게 되므로 *Regression* 문제이다.

예2) 예1의 문제는 원하는 결과를 다음과 바꾸면 *classification* 문제로 바꿀 수 있다. 집의 가격을 예측하는 것이 아니라 *원하는 가격보다 더 비싸게 또는 더 싸게 판매하는 하였는가*로 문제를 바꾸는 것이다. 이 경우에는 2개의 분류로 결과가 나뉘게 된다.

## Unsupervised Learning (비지도학습)
Unsupervised Learning은 우리가 어떤 결과를 얻는 것이 좋은지 알지 못할때 사용하는 방법이다. 데이터의 샘플들이 어떤 연관관계가 있는지 알지 못하는 상태에서 상관관계를 파악할 수 있게 된다.

Unsupervised Learning 문제는 다음과 같이 분류된다.
* clustering
  연관관계가 있는 데이터끼리 묶어내는 문제. 정확한 분류를 알지 못하므로 예측한 결과에 대한 피드백이 없다.
* Associative
  새로운 데이터에 대하여 상관관계가 큰 경우을 추출해서 새로운 데이터에 대한 분류를 하는 문제
