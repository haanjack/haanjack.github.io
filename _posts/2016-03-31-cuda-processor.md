---
layout: post
title:  "CUDA 프로세서에 대한 이해"
date:   2016-03-31 1:00:00
categories: CUDA Programming
description: CUDA 프로그래밍을 위한 Architecture 이해 및 Thread Index 설계
publish: true
categories: CUDA
tags:
- CUDA Programming
- CUDA Architecture
---

지난 CUDA 프로그래밍 포스트에서는 CUDA 프로그래맹 모델에 대하여 이야기 해보았습니다.

이번에는 아래와 같은 주제로 보다 자세히 CUDA 프로그래밍에 대해 알아보도록 하겠습니다.

* <a href="#CUDA Processor Architecture">CUDA Processor 구조</a>
* <a href="#CUDA Processor & Thread">CUDA Thread Hierachy와 CUDA Processor간의 관계</a>
* <a href="#CUDA Thread Index">CUDA Thread Index 구성</a>

맨 마지막 내용이 결론이므로, 컴퓨터 구조에 대한 흥미가 없으시거나 바쁘신 분들은 마지막 내용만 보고 복사해 가셔도 됩니다. 나중에 더 공부하고 싶을때 보셔도 됩니다.

<a id="CUDA Processor Architecture"></a>

# CUDA Processor 구조

CUDA Processor의 구조를 살펴보도록 하겠습니다. 그리고 어떻게 CUDA에서 병렬처리가 이뤄지는지 살펴보도록 하겠습니다.


<img class="col three" src="/images/201603/blockdiagram_big.png"/>
<div class="col three caption">
  Kepler Core Architecture 그림
</div>

위 그림은 CUDA Processor의 구조인데, 가운데 L2 Cache가 보이고 복잡해 보이는 15개의 네모의 집합이 보입니다.
저 네모의 집합을 CUDA Multi-Processor라고 부릅니다. CUDA Multi-Processor는 CUDA Procesor에서 작업을 분할해서 처리하는 Processor로서 CUDA가 작업을 분할해서 처리하는 단위가 됩니다. 즉, CUDA Kernel을 실행했을때 각각의 CUDA Multi-Processor에게 작업이 분할됩니다.

조금 더 상세히 보도록 하겠습니다. 아래 그림은 CUDA Multi-processor의 구조도입니다.

<img class="col two center" src="/images/201603/Kepler.png"/>
<div class="col three caption">
Kepler Multi-prcessor Architecture 그림 (<a href="https://www.youtube.com/watch?v=aAeK19iWda8">네모의 꿈?</a>)
</div>


CUDA Programming에서 최적화의 상당부분은 이 CUDA Multi-Processor에 대하여 얼마나 최적화를 잘하느냐에서 결정됩니다. 따라서 언젠가는 각 요소에 대해서 알고 계셔야 하는데요, 여기서는 간략히 설명을 드리는 정도에서 다음으로 넘어가도록 하겠습니다.

* Processing Units
  * Core
    * Single Point 연산을 수행하는 Processing Unit
    * Multi-Processor에서 많은 수를 차자하며 CUDA가 single floating 연산에 특히 강한 이유가 이 Core가 많기 때문입니다.
  * DP Unit
    * Double Precision Units
    * Floating 연산에서 Double point 연산을 수행하는 Processing Unit 입니다. 정확도를 요구하는 계산에서 활용되며 수가 적은만큼 Kernel 수행시간이 길어집니다.
  * SFU
    * Special Funciton Units
    * sin, cos, sqrt와 같은 특수한 연산에서 사용하는 core입니다.
* Control Units
  * Instruction Cache
    * CUDA Kernel의 명령어를 저장해 놓는 공간
    * CUDA Kernel 실행시 자동으로 Code가 이 부분으로 복사되며, CUDA Multi-Core processor는 개별적으로 이 Cache에서 Instruction을 읽어들여 연산을 수행합니다.
  * Warp Scheduler
    * Warp은 CUDA에서 작업을 처리하는데 기준이 되는 Warp의 Scheduler
    * CUDA Block에서 Warp을 구성하고 CUDA Multi-Processor가 CUDA Block을 성공적으로 수행할 수 있도록 해줍니다.
  * Dispatch
    * CUDA core의 성능을 최대한 높이기 위해서 CUDA Multi-Processor에는 하나의 Warp Scheduler 당 복수의 Dispatch를 갖고 있습니다. Dispatch의 역할은 CUDA Occupancy에 따라 동시에 수행가능한 Warp의 수만큼 CUDA 연산의 병렬성을 극대화 해주는 것입니다. 이를 최적화 하는 방법은 코딩에 정확한 숫자를 입력하기 보다는 CUDA Occupancy의 계산과 실험을 통해 코드를 구성하는 것이며, 최적화를 다루면서 더 상세히 다루도록 하겠습니다.
* Memory Units
  * Register File
    * CUDA Multi-Processor내 필요에 따라 CUDA Core가 나눠서 쓸 수 있는 Register 집합
    * Core에 따라 Register가 독점적으로 구성되어 있는 구조가 아니기에, CUDA Core가 Kernel 수행히 필요한 만큼 Register를 유연하게 Core가 사용할 수 있습니다.
    * Register Dependency에 의한 성능 저하도 최소한으로 줄일 수 있습니다.
  * Shared Memory (L1 Cache)
    * Kernel Code에서 명시된 크기만큼 CUDA Block에서 공유해서 사용 할 수 있는 L1 Cache
    * Core에 독점적이지 않고, Core간에 데이터가 공유 됨
    * Locality가 높은 데이터 또는 Read/Write Buffer 등으로 유용함
  * Read-Only Data Cache
    * 읽기 전용 메모리, 커널 내 공유
    * Look-up Table을 구성하는데 유리
  * Tex
    * 읽기 전용 Pre-fetch Cache
    * CUDA Array를 활용하여 2D 또는 3D로 정렬되어있는 데이터를 효과적으로 읽는데 도움이 됨

이번 포스팅에서는 CUDA Core와 Thread Index간의 관계에 집중해서 설명할 것이므로, 구조에 대해서는 여기까지만 설명할 것입니다. CUDA Memory 구조와 성능에 대해서는 실제 예제를 통해 코딩을 해가면서 실제 사용 방법과 성능을 비교하면서 알아갈 것입니다.

<a id="CUDA Processor & Thread"></a>

# CUDA Thread Hierachy와 CUDA Processor간의 관계

CUDA Processor와 CUDA의 논리적인 구조는 다음과 같은 상관 관계가 있습니다.

| CUDA Architecture | CUDA Logical Architecture |
|---|---|
| CUDA Processor | CUDA Kernel/Grid |
| CUDA Multi-Processor | 복수의 CUDA Block |
| CUDA Core | CUDA Thread |

앞 절에서 CUDA Kernel을 실행했을때 처리해야하는 작업을 CUDA Multi-Processor로 나눈다고 말씀을 드렸었습니다. 이때 작업을 나누는 단위는 CUDA Block의 단위로 나눠지며, 하나의 Multi-Processor는 복수의 CUDA Block들을 처리하게 됩니다. 뿐만 아니라 Warp Scheduler의 도움으로 여러개의 CUDA Block을 각각 순차적으로 처리하도록 할 수 있습니다. 그리고 CUDA Core는 각각 CUDA Thread를 처리합니다.

특히 복수의 CUDA Block이 CUDA Multi-Processor에서 처리된다는 개념이 중요한데, 이는 CUDA Occupancy라는 최적화에서 중요한 개념과 관계가 있기 때문입니다. CUDA Occupance는 최적화에서 상세히 다루기로 하겠습니다.

중요한 점은 CUDA의 하드웨어가 저렇게 계층적으로 구성이 되어 있으므로, CUDA 프로그래밍도 이 계층의 영향을 받는 다는 것입니다. 그것이 CUDA Block이란 개념이 생긴 배경이며, 이를 바탕으로 CUDA Index를 구성하게 됩니다.

<a id="CUDA Thread Index"></a>

# CUDA Thread Index 구성

CUDA Kernel을 개발하실때 주지해야 하는 사항은, 여러분이 짜는 CUDA 코드는 병렬적으로 처리되는 코드이지만, CUDA Thread 입장에서는 오직 하나의 Thread만이 동작하는 코드입니다. 따라서 CUDA Thread는 고유의 Index를 갖고 있으며, 이를 알아내기 위해서는 CUDA Thread Indexing Keyword에 대해서 알고 계셔야 합니다.

CUDA Programming 모델을 설명하면서 CUDA Thread Index를 위한 키워드 들을 잠깐 설명했었습니다.

| 키워드 | 설명 | 차원 |
|---|---|---|
| gridDim | Kernel의 block의 수 | x, y |
| blockIdx | CUAD block의 인덱스 | x, y |
| blockDim | CUDA block의 크기 | x, y, z |
| threadIdx | CUDA 쓰레드의 인덱스 | x, y, z |
| warpSize | CUDA Instruction이 동시에 처리가능한 CUDA 쓰레드 수 | |

위에서 장황하게 CUDA Architecture에 대해서 여러 설명을 하고 그것과 CUDA Block과 CUDA Thread 간의 관계를 설명했습니다. 이는 여기서 설명할 CUDA Thread Indexing의 논리적인 구성을 설명하기 위해서 필요했던 설명이며, 논리적인 구성만 생각해서 이해하셔도 무방합니다.

에를 들어 설명하겠습니다.
10 x 18 크기의 공간을 [4 x 4] 크기의 타일로 덮는다고 했을때 몇개의 타일이 필요할까요? 이때 Tile은 쪼갤 수 없으며, 공간을 덮기만 하면 됩니다.

<img class="col two center" src="/images/201603/Screen Shot 2016-04-10 at 1.54.18 PM.png"/>

이것을 우리는 block의 크기로 먼저 나눌 것입니다.

<img class="col two center" src="/images/201603/Screen Shot 2016-04-10 at 2.01.16 PM.png"/>

단순히 전체 갯수 나누기 block의 크기로 계산하면, $10 \times 18 / 4 / 4 = 11.25$개가 필요하므로 필요한 타일의 수는 11개 또는 12개라고 하실지도 모르겠습니다. 하지만 타일은 쪼갤 수 없다고 하였으므로 가로 5개, 세로 3개로 해서, [10 x 18]의 공간을 [4 x 4] 크기의 타일로 덮으려면 15개의 타일이 필요합니다.

CUDA는 개념적으로 위 그림과 같이, A, B 등으로 라벨링한대로 지정된 CUDA Block의 크기에 맞춰서 데이터를 동시에 처리해줍니다. 그러다보면 C와 같이 데이터 경계를 넘어갈 수도 있습니다. CUDA에서는 이것을 CUDA Index 구성을 통해 우리가 원하는 형태로 CUDA Thread를 데이터에 매핑할 수 있습니다.

만약 타일의 크기가 [8 x 2]라고 한다면 어떻게 될까요. $2 x 9 = 18$개가 될 것입니다.

CUDA의 Grid의 크기를 계산하는 방법도 이와 비슷합니다. 처리해야하는 데이터를 적당한 크기로 자르는 것이죠.
만약 [1920 x 1080] 크기의 데이터를 [16 x 16] 크기의 CUDA Block으로 나눠준다면 총 CUDA block의 수는 120 x 68개가 필요하게 됩니다.

이 경우에 대하여, $width = 1920$, $height = 1080$, $block Width = 16$, $block Height = 16$이라고 할 수 있습니다.
CUDA Keyword로는 다음과 같이 정리됩니다.

| GridDim.x | 68 |
| GridDim.y | 120 |
| BlockDim.x | 16 |
| BlockDim.y | 16 |

여기서 우리는 BlockDim의 값을 [16, 16]이라고 정해줬듯이, Grid Dimension도 정해주어서 CUDA가 적절한 수의 CUDA Block을 생성할 수 있도록 해주어야하며, 데이터마다 CUDA block을 Mapping시켜서 처리하는 경우 다음과 같이 코드를 구성하시면 됩니다. 다만 아래 코드는 식을 보여드리기 위한 것이며, block의 크기는 앞서 CUDA Programming Model에서 설명했던 것 처럼 Kernel를 호출할때 제어 파라미터로 넘겨주셔야 합니다.

{% highlight C %}
grid_dimension.x = (DATA_WIDTH + blockDim.x - 1) / blockDim.x
grid_dimension.y = (DATA_HEIGHT + blockDim.y - 1) / blockDim.y
{% endhighlight %}

위 코드가 복잡하다고 생각이 드시면 ceiling을 하시는 것도 좋습니다만, 같은 결과를 내는 것은 동일합니다.
여하튼 위 코드처럼 하시면 CUDA block의 크기가 정해져 있는 상태에서 데이터의 크기가 변하는대로 CUDA block의 수도 가변적으로 적용되게 됩니다.

`blockIdx` 키워드는 우리가 앞서 쪼갰던 CUDA block의 위치를 의미하며, CUDA Thread는 자신이 속한 `blockIdx`의 값을 알고 있으므로 그냥 참조하시면 됩니다.

`threadIdx`는 CUDA block 내에서 존재하는 CUDA thread의 Index입니다. `blockIdx`와 마찬가지로 각각의 CUDA Thread는 자신만의 CUDA thread Index를 알고 있는데, 이는 상대적인 위치라고 할 수 있습니다. 다만, 이 때문에, CUDA Thread는 어떤 위치에 있는 데이터를 처리해야하는지 알기 위해선 절대적인 인덱스 주소값을 알아야 합니다.

그래서 이전에 CUDA Programming 모델링에 대해서 설명드릴때 아래 표를 보여드리면서 CUDA Thread의 인덱싱은 이대로 한다고 말씀드렸던 것입니다.

| 인덱싱 종류 | 코드 |
|:---:|:--- |
| 1차원 데이터(x only) | blockDim.x * blockIdx.x + threadIdx.x |
| 2차원 데이터(x, y) | blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y |
| 2차원 데이터(x only) | N.x * blockDim.y * blockIdx.y + N.x * threadIdx.y + blockDim.x * blockIdx.x + blockIdx.x |

위의 CUDA Indexing에 대한 예는 데이터 하나당 하나의 CUDA Thread를 매핑한 경우에 해당하는 것이며 필요에 따라서 다양하게 변형이 가능합니다. 만약 이렇게 하지 않으신다면, CUDA Thread 하나 당 데이터를 어떻게 처리할 것인지를 염두에 두고 Index를 수정하시면 보다 쉽게 필요한대로 CUDA Thread Index를 변형해서 사용하실 수 있습니다.

혹시나 질문이 있으시거나 이해가 가지 않는 부분이 있다면 알려주세요.

다음부터는 CUDA 메모리 별로 예제 코드를 갖고 설명을 드리도록 하겠습니다.
