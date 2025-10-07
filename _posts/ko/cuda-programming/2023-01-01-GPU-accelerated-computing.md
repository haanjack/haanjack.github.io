---
layout: post
title: GPU 가속 컴퓨팅
date: 2023-03-10 00:01:00
description: GPU에 기반한 가속 프로그래밍 환경 및 특징에 대해서 살펴봅니다.
tags:
  - cuda
categories: cuda programming
taxonomy: cuda
published: false
permalink: /ko/gpu-accelerated-computing
translation: /en/gpu-accelerated-computing
---



오늘날 우리는 일상생활해서 수 많은 프로그램을 이용하면서 살아가고 있습니다. <br/>
이 프로그램들은 우리가 아침에 일어나 아침식사를 위해 냉장고 문을 여는 순간부터,
잠들기 전 유투브를 시청하는 순간에 이르기까지 우리는 다양한 프로그램을 쉽게 접하고 이용하고 있습니다.

이러한 프로그램들은 대부분의 경우 정해진 순서와 조건에 따라 동작하도록 고려되어 설계되어 있습니다. <br/>
이들이 구동되는 CPU는 대부분의 프로그램에서 요구하는 연산량을 뒷받침하기에 충분할 정도로 발전이 되었습니다.

그렇지만 데이터의 폭발적인 증가와 이를 합리적인 시간 내에 처리하고자 하는 수요 또한 증가하였습니다. 이러한 수요를 충족시키기 위해서 컴퓨팅 환경은 병렬처리나 분산처리와 같은 기술이 고안되었습니다. GPU 프로그래밍은 이 중 그래픽스 분야의 빠른 처리를 위해 발전하기 시작하여, CUDA와 같은 개발언어의 발달에 힘입어 다양한 분야에 접목되기 시작한 분야입니다.

이러한 GPU 프로그래밍은 절차적으로 수행하는 프로그램에서 연산량이 많이 필요한 부분에 대하여 GPU가 가진 장점을 활용하여 프로그램의 수행시간을 향상시키는 것을 목적으로 하고 있습니다. 이를 도식적으로 나타내면 다음 그림처럼 설명할 수 있습니다.

GPU가 CPU와 비교하여 연산성능에서 장점을 갖는 이유는 더 많은 Core를 바탕으로 효율적인 연산이 가능하도록 설계되어 있기 때문입니다. 간단한 그림을 통해 이해해 보도록 하겠습니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/cuda/what-a-gpu-does-pic-768x489.png" title="What a gpu does" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    (출처: NVIDIA blog)
</div>

이 둘은 엄연히 목적이 다르기에 구조적으로 많은 차이를 갖고 있지만, 특히 연산을 수행하는 코어 수에서 많은 차이가 있음을 알 수 있습니다. 이러한 병렬처리 구조가 갖고 있는 장점은 어떤 것이 있을까요? 다음의 영상은 CPU와 GPU의 구동방식의 차이를 쉽게 이해할 수 있도록 보여줍니다.

<div class="row">
    <div class="col-sm mt-1 mt-md-0">
        <iframe width="560" height="420" src="https://www.youtube.com/embed/-P28LKWTzrI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </div>
</div>

위 영상은 유쾌하게 둘 차이를 표현했지만 사실 많은 의미를 함축하고 있습니다. CPU는 순차적으로 처리하는 구조를 갖고 있지만 GPU는 동시에 처리할 수 있는 하는 양이 CPU대비 월등히 높다는 점입니다. 또한 이러한 연산 성능을 뒷받침하기 위해서 더 많은 데이터를 처리할 수 있는 메모리 구조를 가지고 있습니다.

그러나 그렇다고 해서, GPU가 CPU로부터 독립적으로 연산을 수행하는 것은 아닙니다. 여타 프로그램과 같이 GPU의 동작도 절차에 따라 수행되는 가운데 필요한 부분에 대하여 GPU의 연산성능을 활용해야 합니다. 아래 그림은 GPU를 활용한 컴퓨팅에 일반적인 절차를 보여주는 예입니다.

![](https://researchcomputing.princeton.edu/sites/g/files/toruqf311/files/styles/freeform_750w/public/2021-11/gpu_as_accelerator_to_cpu_diagram.png?itok=q0YaEuYH)

위 그림의 설명과 같이 GPU 프로그래밍은 CPU 프로그래밍의 일부로서 동작하게 됩니다.

### 다양한 프로그래밍 언어

GPU 프로그래밍을 하기 위한 언어는 CPU의 언어가 다양한 것과 마찬가지로 매우 다양합니다.

각 GPU 제조사에서는 고유의 언어를 개발하여 제공하고 있습니다. 대표적인 예로는 NVIDIA의 [CUDA](https://developer.nvidia.com/cuda-toolkit)나 AMD의 [HIP](https://developer.amd.com/resources/rocm-learning-center/fundamentals-of-hip-programming/)이 있습니다.

사실 위와 같이 병렬로 연산이 가능한 알고리즘에 대하여 코어를 병렬로 제어하는 방식의 프로그래밍 모델은 병렬처리를 할 수 있는 모든 하드웨어에서 활용할 수 있습니다. 따라서 이런 방식에 기반하여 여러 종류의 플랫폼을 공통적으로 지원하기 위한 언어들([OpenACC](https://www.openacc.org/), [SyCL](https://www.khronos.org/sycl/), [OpenCL]())이 개발되어 오고 있습니다. 또는 [xla](https://github.com/openxla/xla)와 같이 machine learning을 목적으로 하는 경우도 있습니다.

또한 Python을 기반으로 GPU 프로그래밍 모델을 지원하는 framework 또한 다수 공개가 되어 있습니다.
 - [cupy](https://cupy.dev/)
 - [Numba](https://numba.pydata.org/numba-doc/latest/cuda/index.html)
 - [pycuda](https://documen.tician.de/pycuda/)
 - [triton](https://openai.com/research/triton)

따라서 개발언어를 선택할때, 목적과 환경에 맞게 선택해서 사용하듯이 GPU 개발언어 역시 선택하여 사용할 수 있습니다. 다만 GPU를 더욱 효과적으로 활용하기 위해서는 어느정도는 GPU의 동작 특성을 이해하고 있는 것이 좋을 것입니다.


