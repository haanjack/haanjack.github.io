---
layout: post
title:  "CUDA 프로그래밍 모델"
date:   2016-03-27 1:00:00
categories: CUDA Programming
description: CUDA 프로그래밍을 위한 기초 개념
publish: false
categories: CUDA
tags:
- CUDA Programming
---

CUDA 프로그래밍을 하면서 제가 겪었던, 또는 주위에 CUDA 프로그래밍을 주변에 소개하면서 겪었던 가장 큰 어려움은 CUDA 프로그래밍의 개념을 이해하거나 이해시키는 일이었습니다.

일반적인 병렬처리 기법으로 많이 사용하는 멀티쓰레드나 멀티프로세싱, 그리고 그것들의 온전한 동작을 위해 운영체제에서 제공하는 다양한 도구들은 그나마 개념적으로나마 이해할 수 있었지만, CUDA는 아무래도 다른 하드웨어에서 다른 동작 조건을 갖고 동작하기 때문인 것 같습니다. 서너개의 병렬처리를 상상하기도 벅찬데 수백개의 멀티쓰레딩을 할 수 있다는 것은 장점으로 느껴지다가도 다시 한번 생각하면, 그걸 어떻게 하는 건지 겁이 나는 것도 사실입니다.

때문에 CUDA Programming Model은 CUDA 프로그래밍을 하는데 있어서 중요한 개념을 제공해 줍니다.

* <a href='#SIMT'>CUDA Processing 모델 (SIMT Architecture)</a>
* <a href='#Kernels'>CUDA Kernels</a>
* <a href="#Thread Hierachy">CUDA Thread Hierachy</a>
* <a href="#Memory">Memory 계층</a>
* <a href="#HC">Heterogeneous Computing</a>
* <a href="#Compute Capability">Compute Capability</a>

본격적으로 CUDA 프로그래밍을 시작하기에 앞서서 주요 개념들을 살펴보도록 하겠습니다.

<br>

<a id='SIMT' class='anckor'></a>

# SIMT (Single Instruction Multiple Thread)
SIMT라는 개념은 nVidia에서 자신의 CUDA 동작을 설명하기 위해서, CPU에서 사요하는 용어를 차용해서 만든 조어입니다. CPU에서는 주로 SIMD(Single Instruction Multiple Data)라는 용어를 사용하는데, CPU의 성능을 최대한 활용하기 위해서 하나의 명령어로 여러개의 데이터를 처리하도록 하는 동작을 의미합니다. 이와 비슷하게 CUDA는 하나의 명령어로 여러개의 쓰레드를 동작시킵니다. 그러면 각각의 CUDA 쓰레드는 데이터를 하나씩 처리하지만, CUDA에서 여러개의 쓰레드를 동작시키므로 동시에 여러개의 데이터를 처리할 수가 있습니다.

즉, 용어와 방법은 다르지만 SIMD와 SIMT는 동일한 목적인 데이터를 병렬로 처리한다는 개념을 갖고 있습니다. 이는 앞으로 CUDA 프로그래밍을 하는데 있어서 중요하면서도 기본적인 개념이므로 새로운 개념을 접하시거나 코딩하실때 염두에 두고 개발하셔야 합니다.

<a id="Kernels" class='anckor'></a>

# Kernels
Kernel은 GPU에서 처리하는 동작 내지는 함수를 의미하며, GPU가 처리하는 function 코드라고 볼 수 있습니다. 즉, 앞서 소개한 SIMT 개념을 담고 작성된 CUDA 코드입니다.

예제를 통해서 살펴보도록 하겠습니다.
C를 기반으로 하므로, C 언어를 기준으로 보셔야 합니다.

{% highlight C linenos %}
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
...
    // Kernel invocation with N threads
    VecAdd<<<M, N>>>(A, B, C);
...
}
{% endhighlight %}

위 코드에서 `__global__` 키워드가 달려있는 함수인 `VecAdd`를 kernel이라고 부릅니다.
한편, line 11의 `Vector<<<M, N>>>(...)`는 커널을 호출하는 라인입니다. 각 부분이 구체적으로 어떤 것을 의미하는지는 다음 절에서 설명드리도록 하겠습니다.

이와 비슷한 CUDA Keyword로는 `__device__`, `__host__` 등이 있습니다. 각각이 의미하는 바는 다음과 같습니다.

| 키워드 | 의미 |
| --- | --- |
| __global__ | CPU Main thread에서 호출 가능한 GPU에서 동작하는 함수 |
| __host__ | CPU Thread에서만 동작하는 함수 |
| __device__ | GPU 커널함수 내부에서만 호출 가능한 GPU 함수 |
| __host__ __device__ | CPU/GPU 함수 각각 모두 호출 가능한 함수 |

이러한 CUDA 함수 키워드들은 CUDA 컴파일러로 하여금 코드의 동작 대상을 알려주어 개발자가 원하는 동작을 할 수 있도록 도와주는 한편, 개발하는 분들에게도 함수가 어떤 동작조건으로 동작하는지 알려주는 역할을 한다고 할 수 있겠습니다.

중요한 점은, CUDA 프로그래밍은 커널이라는 개념을 GPU에서 동작하는 함수라고 부르며, 이를 기준으로 CPU와 GPU 동작을 구분한다는 것입니다. 그래서 같은 파일에 있는 소스코드이지만 키워드를 기반으로 동작하는 하드웨어는 다르도록 개발 할 수 있습니다.

<a class='anckor' id="Thread Hierachy"></a>

# Thread Hierachy
앞서 설명드린대로 CUDA는 SIMT 구조를 갖고 있으며, 여러개의 Thread를 동작시킵니다. 다만 스스로 알아서 하는 것은 아니고 얼마나 많은 쓰레드를 동작시켜야만 하는지 알려주어야만 합니다.

이때 알아야 하는 것이 Thread Hierachy입니다.

<img src="/images/201603/CUDA grid and block.png" width="400px" />

그림을 보시면 CUDA Thread는 CUDA Block 또는 CUDA Thread Block으로 묶여서 동작합니다. 그리고 CUDA Block은 다시 Grid라는 단위로 묶여서 동작합니다.

앞에서 사용한 예제 코드를 다시 확인해 보도록 하겠습니다.

{% highlight C linenos %}
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}
int main()
{
...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
{% endhighlight %}

`line 5,6`는 CUDA 쓰레드의 인덱스를 변수로 지정하는 라인이며, `line 7`은 앞에서 알게된 쓰레드 인덱스로 연산을 수행하는 부분입니다. 여기서 CUDA 쓰레드는 앞서 설명드린 SIMT 구조에서 말하는 쓰레드이며, CUDA에서는 각각의 쓰레드를 구별할 수 있도록 `threadIdx` 구조체를 제공합니다. 그래서 CUDA 쓰레드는 자신의 index를 동작 중에 알 수 있게 되고, 어떤 데이터를 처리할 것인지 인덱스 정보를 바탕으로 판단하게 됩니다.

그렇다면 몇 개를 동작시킬 것인지는 CUDA 커널은 어떻게 알 수 있을까요?
`line 80`에 있는 `<<<numBlocks, threadsPerBlock>>>`은 그 답을 알고 있습니다.
위 코드를 직역하자면 `블럭 하나에, thread는 NxN개를 실행시켜라`라는 것인데, `<<< ... >>>`을 통해 CUDA 커널의 제어 파라미터를 전달해 주는 것입니다.

`<<< ... >>>`에는 CUDA block의 개수와 Block Size(CUDA Thread의 개수)를 입력해줍니다. 위 예제의 경우에는 block 개수는 1, thread의 수는 NxN으로 정해놓은 것입니다. 그렇다면 NxN개는 말 그대로 아무크기나 지정해도 되는 것일까요?

실제로는 그렇지 않으며, CUDA 하드웨어 스펙에 의해 최대 크기가 정해져 있고, 최적 사이즈도 대체적으로 정해져있는 편입니다. 제 경험상 `NxN = 256`인 경우가 보통은 최적 크기입니다. 즉 `(16)x(16)`이라는 이야기인데, 이것은 그리 크지 않은 크기입니다. 그렇다면 그 이상의 데이터를 처리해야하는 경우에는 어떻게 해야할까요? 쉽게 생각하면, 우리가 만든 CUDA block을 다른 데이터를 지정해서 여러번 실행시키면 될 것입니다. CUDA에서는 이것을 Grid라는 개념으로 동작을 하며, `gridDim`을 통해서 지정할 수 있습니다. 즉, 몇 번 block이 재사용되어야 하는지 CUDA 커널에 알려주고, CUDA는 위 그림에서 Grid와 같은 형태로 Block을 전개하고, block 내부에서는 여러개의 Thread를 동작시키는 것입니다.

이를 나타낸 예제를 갖고 설명을 드리겠습니다. 이 예제는 매우 큰 `NxN`개의 데이터를 CUDA에서 처리하는 동작입니다.

{% highlight C linenos %}
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}
int main()
{
...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
{% endhighlight %}

`line 14`를 보시면 제가 조금전에 설명드린대로 일반적인 최적 사이즈라고 부르는 대로 `[16, 16]`크기의 thread block을 구성했습니다. 그리고 numBlocks의 크기는 N을 thread block의 크기로 나누어 CUDA block이 몇 번 실행되어야 하는지 계산을 했습니다. (정확한 계산은 좀 다르지만 이 부분은 차후에 더 설명을 드리겠습니다.) 이떄 x와 y에 대해서 각각 계산을 해주어서 2차원 인덱스를 구성했다는 점입니다.

이어서 CUDA Kernel을 살펴보도록 하겠습니다. `line 5,6`을 보시면 처음보는 키워드 들이 들어가 있습니다. `blockIdx`, `blockDim`은 CUDA Thread block의 인덱스와 크기를 나타내는 키워드입니다. 특히 `blockDim`은 앞서 CUDA Kernel를 호출할때 `line 16`에서 ```<<< ... >>>```에 전달해준 제어 파라미터 중 threadPerBlock와 동일한 값을 갖습니다. 즉, 어떻게 `NxN` 크기의 데이터를 쪼개서 처리할 것인지 CUDA로 하여금 알 수 있게 해주는 키워드입니다. 이것에 인덱스 값을 추가로 계산하면 CUDA Thread의 Index를 알 수 있게 되고, 처리할 데이터의 인덱스를 알 수 있게 됩니다.

이렇게 복잡하고 다양한 인덱싱을 지원하기 위해서 CUDA는 총 5가지 키워드를 제공하고 있습니다.

| 키워드 | 설명 | 차원 |
|---|---|---|
| gridDim | Kernel의 block의 수 | x, y |
| blockIdx | CUAD block의 인덱스 | x, y |
| blockDim | CUDA block의 크기 | x, y, z |
| threadIdx | CUDA 쓰레드의 인덱스 | x, y, z |
| warpSize | CUDA Instruction이 동시에 처리가능한 CUDA 쓰레드 수 | |

즉, 크기를 알려줘야하는 Grid와 Block에는 `-Dim`이라는 이름의 키워드가, index를 알아야만 하는 block과 thread에는 `-Idx`라는 이름의 키워드가 있는 것입니다.

이를 잘 조합한다면 CUDA Kernel 코드는 인덱싱만 잘해도 하나의 쓰레드만 처리하는 코드를 짜도 병렬처리가 가능하게 됩니다. 앞서 설명 드렸던 SIMT 구조가 이렇게 드러나는 것이죠.

인덱싱에 대한 대표적인 예를 나열해 보겠습니다.

| 인덱싱 종류 | 코드 |
|:---:|:--- |
| 1차원 데이터(x only) | blockDim.x * blockIdx.x + threadIdx.x |
| 2차원 데이터(x, y) | blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y |
| 2차원 데이터(x only) | N.x * blockDim.y * blockIdx.y + N.x * threadIdx.y + blockDim.x * blockIdx.x + blockIdx.x |

위 표에서 2번째와 3번째는 2차원 인덱스를 한다는 점에서는 동일하지만 위의 것은 2차원 배열을 사용해야만 하고, 3번째는 1차원 배열에 저장된 2차원 데이터에 대하여 2차원 인덱싱을 해준다는 점에서 차이가 있습니다. 세번째가 복잡해서 다음과 같이 변경해서 사용할 수도 있습니다.

> idx_x = blockDim.x * blockIdx.x + threadIdx.x
> idx_y = blockDim.y * blockIdx.y + threadIdx.y
> idx = idx_y * N.x + idx_x

따지고보면 표의 세번째와 같은 연산임을 알 수 있는데, 정확히 뭘 하려는 것인지 코드상으로 알 수 있고 실수해도 바로 알 수 있다는 장점이 있습니다.

끝으로, Warp이라는 용어가 위 표에서 나왔었는데요, 표에 있는대로 Warp은 CUDA에서 동시에 처리가능한 또는 동시에 처리하는 CUDA Thread 수 입니다. 병렬처리 연산을 수행하는데 Warp을 단위로 작업을 Scheduling하고 Warp 개수 만큼을 병렬처리 할 수 있습니다. Warp의 크기는 CUDA Compute Capacity에 따라서 다를 수 있으며 보통은 32개 입니다. 이 크기는 가변적으로 설정할 수는 없습니다. 다만 코드의 내용에 따라 Warp을 충분히 활용하지 못해서 CUDA Thread가 대기상태로 놓일 수는 있으며 이를 따지는 것이 CUDA Occupancy입니자만, 이부분은 별도의 포스트에서 다루도록 하겠습니다.

제가 드린 설명이 이해가 되시나요? ^^;
CUDA 프로그래밍을 하시는데는 이정도면 충분하지만, 정확한 이해를 위해서는 CUDA Processor에 대한 설명도 필요한데요, 이부분에 대해서는 별도의 포스팅으로 설명을 드리도록 하겠습니다.

<a class='anckor' id="Memory"></a>

# Memory 계층

CUDA에서 사용하는 메모리 계층은 크게 두 가지로 구별 할 수 있습니다.

1. Global Memory
 - 그래픽 카드에 설치되는 메모리. 1G이상. 다른 메모리에 비해 느리다. 물론 CPU에서 사용하는 메모리보다는 빠르다. 대역폭에 신경을 써야 한다.
1. On-Chip memory
 - Shared L2 Cache: CUDA Core간 공유하는 Cache 메모리. 프로그래밍에 노출되지 않음.
 - Shared L1 Cache: CUDA Core 내에서 사용하는 L1 Cache. (약 64KB. 코어 버전마다 다름). 프로그래밍 가능
 - Texture Memory: CUDA Core가 공유하는 Pre-fetch를 지원하는 read-only cache. 프로그래밍 가능.
 - Constant Memory: CUDA Core가 공유하는 Kernel 실행중 업데이트가 불가능한 read-only cache. 프로그래밍 가능.
 - Register File: CUDA Thread가 사용하는 Register

각각의 메모리는 개별적인 설명이 필요한데, 이 부분을 잘 이해하는 것이 CUDA 프로그래밍을 최적화 시켜나가는데 중요한 부분이라고 할 수 있습니다. 아래 그림은 CUDA Core의 그림입니다. 위에서 설명 드린 것들이 곳곳에 있습니다만, 보이시나요? ^^;;

아래 정사각형 그림이 CUDA Multiprocessor라고 불리는 것이고, 위는 그것이 여러개가 모여있는, 우리가 흔히들 GPU 코어라고 부르는 것의 내부 모습입니다.

<img src="/images/201603/blockdiagram_big.png" width="600"/>

<img src="/images/201603/Kepler.png" width="300"/>

이 그림과 CUDA Thread간의 상호작용을 하는 것이 CUDA 프로그래밍인데요, CUDA 하드웨어의 구조와 CUDA Thread간의 관계, 그리고 메모리가 연동되는 부분은 별도의 포스팅으로 설명을 드리도록 하겠습니다.

<a class='anckor' id="HC"></a>

# Heterogenious Computing

Heterogeneious Computing은 예전에 하나의 코어에서 연산을 모두 처리하던 것에서 별도의 연산을 위한 하드웨어를 통해서 컴퓨터의 처리성능을 높이는 컴퓨팅 방식을 말합니다. DSP, FPGA 등을 활용한 것들이 예를 들 수 있겠는데요, CUDA 또한 CPU의 제어 하에서 고성능의 연산을 CPU 대신 처리해 준다는 점에서 비슷하다고 할 수가 있겠습니다.

CUDA에서는 CPU에서 사용하는 Host 메모리와 그래픽 카드에 있는 Global (Device) Memory간의 데이터 전송 및 Kernel 호출 등, CUDA의 올바른 동작을 위해 필요한 절차 등을 의미합니다.

<a class='anckor' id="Compute Capability"></a>

# Compute Capability

Compute Capability는 CUDA 코어의 하드웨어 버전을 의미하며, CUDA 개발을 위해 여러분이 설치하셨을 CUDA 버전과는 별개의 것입니다. 버전에 따라서 캐시의 크기와 코어의 개수 등 여러 부분에 대해서 성능과 기능이 개선되어 왔으며, 이에 따라 최적화 결과가 조금씩 달라질 수 있습니다.

이것에 관해서 한가지 말씀드릴 수 있는 것은, 성능과는 별개로 버전이 높을수록 최적화 문제 등에 여러분이 개발에서 겪을 어려움을 많이 줄일 수 있을 것이란 것입니다.

CUDA는 현재 버전이 7.5이며, Compute Capability 버전은 5.3입니다. 호환성 문제를 생각해볼떄, 여러분의 SW를 배포할 경우 CUDA 버전은 고려하지 않으셔도 되지만, Compute Capability는 고려하셔야 합니다. 만약 여러분이 최신의 Compute Capability만을 지원한다면 여러분의 SW를 배포받은 분들은 지원이 되지않을 확률이 아무래도 높겠죠. 왜냐하면 여러분은 최신의 그래픽카드를 사용하지만, 대부분의 사람들은 한번 사놓고 몇년간 하드웨어를 교체하지 않으니까요.

그보다도 더 문제는... nVidia가 CUDA Compute Capability 를 빠른 시일내에 올리고 있고, 제가 가지고 있는 GPU가 얼마 되지도 않아서 구형이 되어버린다는 것이겠죠. 예를 들어, Google의 Deep Learning 라이브러리인 Tensorflow는 CUDA Compute Capability의 최 하위 버전이 3.5 입니다. 비공식적으로 3.0부터 지원한다고는 하지만 개발자가 테스트를 하지 않았다는 것은 얼마든지 문제가 발생할 소지를 가지고 있는 것이니, 신경을 쓰셔야 합니다.

CUDA Compute Capability에 대한 상세한 스펙은 [CUDA Wikipedia](https://en.wikipedia.org/wiki/CUDA#Supported_GPUs)를 참고하시기 바랍니다.
