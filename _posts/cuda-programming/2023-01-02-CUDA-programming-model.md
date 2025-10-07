---
layout: post
title: CUDA 프로그래밍의 기본 특징
date: 2023-03-10 00:00:00
description: CUDA 프로그래밍의 기본적인 개념과 특징에 대해서 살펴봅니다.
tags:
  - cuda
categories: cuda programming
taxonomy: cuda
published: false
translation: /en/cuda-programming/cuda-programming-model
permalink: /ko/cuda-programming/cuda-programming-model
---

# CUDA Programming의 기본 특징

연산 가속기로서 GPU를 활용하는 CUDA 프로그래밍은 다음과 같은 특징을 가지고 있습니다.

- Kernel 프로그래밍
- 이기종 프로세싱 (Heterogeneous Processor Computing)
- 계층적인 쓰레드 구성과 메모리 구조
- Single Instruction Multi-Thread
- CUDA 프로그래밍 언어의 버전과 하드웨어 버전간의 하위 호환성 (Compute Capability)

이러한 특징들은 CUDA 프로그래밍의 특징을 규정하는 몇가지 주요 요소들 중 하나이며, GPU의 동작을 이해하고 효과적으로 활용하는데 있어서 출발점이라고 할 수 있습니다. 이러한 특징들을 모아 CUDA 프로그래밍 모델이라고도 부릅니다.

이 글에서는 각각에 내용에 대해서 자세히 살펴보도록 하겠습니다.

## Kernel 프로그래밍

GPU에서 동작시키는 것을 목적으로 작성하는 함수를 CUDA 프로그래밍에서는 Kernel Function(또는 함수)라고 부릅니다.

이 커널 함수들의 기본적인 생김새는 C/C++ 언어와 비슷하게 생겼지만, 몇가지 주요 CUDA 키워드를 통해 CUDA 컴파일러로 하여금 GPU에 동작하는 함수임을 알리는 한편 세부동작들을 정의할 수 있습니다. 커널함수에서 정의되는 내용은 GPU에서 동작하는 코드이기 때문에 병렬처리를 목적으로 설계된 알고리즘을 코드로 작성해야 합니다. 따라서 CUDA 프로그래밍을 하고자 하는 개발자는 GPU에서 수행하고자 하는 동작을 명확히 구분하는 것은 물론 병렬성을 갖는 코드를 작성할 수 있어야 합니다.

## 이기종 프로세싱 (Heterogeneous Processor Computing)

컴퓨터 시스템에서 CPU 코어 이외의 다른 종류의 프로세서를 추가하여 프로그래밍하는 방식을 우리는 이기종 프로그래밍이라고 부릅니다. 여기서는 CUDA를 활용할 수 있는 NVIDIA GPU에 특정한다고 할 수 있겠습니다. 각각의 프로세서는 독자적인 프로그래밍 모델을 갖고 있으며, 고유의 메모리를 가지고 있을 수 있습니다. 그러나 이들은 하나의 컴퓨팅 시스템안에 속해서 시스템의 목적에 맞게 데이터를 공유하고, 상호간의 동작 순서에 대하여 정해야 하는 등 시스템적으로 흥미로운 스케줄링을 해야 합니다.

## 계층적인 쓰레드 제어와 메모리 구조

대부분의 프로세서가 그러하듯이 GPU에서도 레지스터와 L1/L2 캐시 등을 가지고 있습니다.

- Kernel 중심 프로그래밍
    - 데이터 병렬처리
- 계층적인 쓰레드 제어와 메모리 구조
    - 하드웨어의 구조적인 특징
- 병렬성과 동시성의 혼합
    - Low Latency vs higher throughput
    - 하나의 명령어로 여러 쓰레드를 실행 (SIMT: Single Instruction Multiple Threads)
- CUDA 프로그래밍 언어의 버전과 하드웨어 버전간의 하위 호환성 (Compute Capability)

## Single Instruction Multiple Threaad 동작

CUDA의 동작의 중요한 특징 중 하나는 SIMT (Single Instruction Multiple Thread) 동작을 한다는 것입니다. 이것은 하나의 명령어에 대하여 여러개의 병렬처리가 되는 것을 말하는데, CPU에서 하나의 명령으로 여러개의 데이터를 처리하는 SIMD(Single Instruciton Multiple Data)나 동일한 명령을 수행하는 여러개의 Thread가 동작하는 멀티 쓰레딩(Multi-threading)과는 차이가 있습니다.

SIMT 방식을 통해서 CUDA에서는 병렬로 데이터를 처리하는 한편 각기 독립적으로 동작할 수 있는

## Compute Capability