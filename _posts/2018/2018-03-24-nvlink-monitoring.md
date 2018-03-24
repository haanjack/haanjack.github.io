---
layout: post
title:  "NVLINK monitoring"
description: "NVLINK가 있는 장비에서 nvlink 사용 확인하기"
date: 2018-03-24
published: true
categories: gpu
tags:
 - nvlink
 - nvidia
---

nvlink가 지원되는 서버에서 nvlink를 이용한 통신량을 확인하기 위해선 아래와 같은 명령으로 간단히 확인할 수 있다.

## 1. nvlink counter 활성화
{% highlight bash %}
$ nvidia-smi nvlink -sc 0bz
{% endhighlight %}
  - 0 번 카운터에 대하여
  - byte 정보를 보도록 하며
  - all traffic에 대해 측정

## 2. counter 결과 보기
{% highlight bash %}
$ nvidia-smi nvlink -g 0
{% endhighlight %}
  - 0번 카운터에서 측정된 수치 보기
  <img class="col two center" src="/images/2018-03-24-nvlink-count-1.png"/>

1번과 2번 사이에 필요한 application을 동작시키면 GPU간 nvlink를 통한 peer-to-peer 통신량을 확인할 수 있다.

## 3. Counter 초기화
아래의 명령으로 counter의 값을 초기화 할 수 있다.
{% highlight bash %}
$ nvidia-smi nvlink -r 0
{% endhighlight %}

## 4. Counter 닫기
반대로 counter를 닫는 방법은 다음과 같다.
{% highlight bash %}
$ nvidia-smi nvlink -sc 0bn
{% endhighlight %}

## 5. Counter로 할 수 있는 것
counter를 이용해서 할 수 있는 것들로는 다음과 같은 것들이 있다.
  - application의 nvlink 활용 확인
  - application의 통신량 확인. application 실행 전에 clear 할 것.
  - 현재 nvlink의 bandwith 확인. 단, 주기적으로 clear하면서 모니터링 해야한다.
  - 시스템에서 총 nvlink 통신량 측정

Counter의 수는 총 두 개(0, 1)이므로, 위 가능한 것 중 목적에 맞게 두 가지를 조합해서 사용하면 된다.
