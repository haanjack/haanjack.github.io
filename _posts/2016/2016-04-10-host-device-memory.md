---
layout: post
title:  "CUDA GPU 메모리"
date:   2016-04-10 4:18:00
description: CUDA GPU 메모리의 종류 및 활용법
publish: false
categories: CUDA
tags:
 - CUDA Programming
---

GPU에서 동작하는 메모리의 종류 및 활용 방법에 대해서 알아보도록 하겠습니다. 이 포스트는 다음과 같은 것들을 다룰 예정입니다.

* <a href="#device memory">Device 메모리</a>
* <a href="#shared memory">Shared 메모리</a>
* <a href="#constant memory">Constant 메모리</a>
* <a href="#texture memory">Texture & Surface 메모리</a>

<a id="device memory" class="anckor"></a>

# Device 메모리

CUDA의 메모리에 대해서 알아보기 전, 먼저 CUDA가 동작하는 환경에 대해서 간략히 짚어보도록 하겠습니다.
앞으로 개발할 환경은 HC(Heterogeneous Computing)또는 HPC(Heterogeneous Processor Computing)라고 불리는, 한가지 이상의 프로세서를 이용하여 컴퓨팅을 이용한 환경입니다. 이는 보통의 프로그램을 개발하는 것과 독립된 동작 환경을 갖고 있는 임베디드 프로그래밍하고는 또 다른 형태의 프로그래밍 개념인데, 보조 연산 프로세서가 있다고 생각하시면 됩니다. 아래 그림을 보고 설명 드리겠습니다.

<img class="col two center" src="/images/201604/CUDA_processing_flow_(En).png"/>
<div class="col three caption">
CPU와 GPU 동작 순서. (Source: Wikipedia)
</div>

위 그림은 CUDA 코어의 동작 순서를 설명하고 있습니다. 먼저 이 그림의 구성요소를 살펴보면, 두 개의 processing unit이 있고, 두 종류의 Memory가 있습니다. 여기서 각각의 Processing Unit은 자신이 제어할 수 있는 메모리를 독립적으로 갖고 있으며, 서로 제어할 수는 없습니다. 따라서 CUDA에서 구현된 GPU 코드는 GPU 메모리만을 읽고 쓸 수가 있으므로, GPU 코드를 동작하기 전과 후에 GPU 메모리로 데이터를 전송하고 결과를 가져오는 절차가 있어야 합니다. 다음 절에서 이 부분에 대해서 CUDA가 어떤 방법들을 제공하고 있는지 살펴보도록 하겠습니다.

<div class="img_row center">
<img class="col one" src="/images/201604/cpu.jpeg"/>
<img class="col one" src="/images/201604/card-front.jpg"/>
</div>
<div class="col two caption center">CUDA는 Host(CPU)와 Device(GPU)를 구분하여 개발합니다.</div>

따라서 CUDA에서는 CPU와 GPU를 분리해서 생각하며, CPU를 Host, GPU를 Device라고 부릅니다. 이는 CPU 입장에서 GPU 역시 보조 프로세서이고, 데이터는 기본적으로 CPU 쪽 메모리에 있기 때문에 주인(Host)입장이기 때문입니다. 이와 비슷하게 CPU를 위한 메모리는 Host 메모리, GPU 메모리는 Device 메모리라고 합니다.

## 할당 및 초기화

CUDA Host 메모리 할당은 기존의 메모리 할당과 크게 차이가 없습니다.

{% highlight C %}
float* h_A = (float*)malloc(size);
{% endhighlight %}

Device 메모리의 할당은 `cudaMalloc`을 사용하여 할당을 합니다.

{% highlight C %}
float *d_A;
cudaMalloc(&d_A, size);
{% endhighlight %}

그러면 **h_A**와 **d_A**는 같은 사이즈이지만 각기 **Host**와 **Device**에 메모리를 할당하게 됩니다. 그럼 이를 바탕으로 예제를 살펴보도록 하겠습니다.

## CUDA 메모리 해제

그러하면 GPU 메모리에 할당한 Device 메모리는 어떻게 해제할까요. 이를 위해서 CUDA에서는 **cudaFree**라는 함수를 제공하고 있습니다.

`cudaFree(void* p_mempry);`


## 데이터 복사

CUDA 함수를 실행하기 이전에 또는 CUDA 동작의 결과물을 가져오기 위해서 CUDA 메모리를 복사하는 과정이 반드시 필요하다고 앞에서 말씀드렸었습니다. 이를 위해서 CUDA 에서는 다음과 같은 함수를 제공하고 있습니다.

`cudaMemcpy(void* p_dst, void* p_src, int size, int direct)`

| Direct | 내용 |
| memcpyHostToDevice | memcpy Host -> Device |
| memcpyDeviceToHost | memcpy Device -> Host |
| memcpyDeviceToDevice | memcpy Device -> Device |
| memcpyHostToHost | memcpy Host -> Host |

cudaMemcpy함수는 CUDA에서 제공하는 대표적인 메모리 복사 함수입니다. 복사하는 방향을 지정해주는 미리정의된 파라미터가 있다는 것이 특징입니다. 이에 따라서 메모리 주소를 함수에 전달해줄때 어떻게 써야하는지도 신경써야 합니다. 아래 코드를 보시면 이해하실 수 있습니다.

{% highlight C linenos %}
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
{% endhighlight %}

Host 메모리에 있는 데이터를 Device 메모리로 복사하게 되는 경우, cudaMalloc으로 할당한 메모리를 dst로, Host 메모리를 src로 전달합니다. 역으로 Device 메모리의 데이터를 Host 메모리로 가져오는 경우에는, 그 반대로 해주어야겠죠. 헷갈리는 것 같으면서도 간단한것 같습니다. 그럼 간단한 예제로 확인해보겠습니다.

{% highlight c linenos %}
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

// Host code
int main()
{
int N = ...;
  size_t size = N * sizeof(float);
  // Allocate input vectors h_A and h_B in host memory
  float* h_A = (float*)malloc(size);
  float* h_B = (float*)malloc(size);
  // Initialize input vectors
...
  // Allocate vectors in device memory
  float* d_A;
  cudaMalloc(&d_A, size);
  float* d_B;
  cudaMalloc(&d_B, size);
  float* d_C;
  cudaMalloc(&d_C, size);
  // Copy vectors from host memory to device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  // Invoke kernel
  int threadsPerBlock = 256;
  int blocksPerGrid =
          (N + threadsPerBlock - 1) / threadsPerBlock;
  VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  // Free host memory
... }
{% endhighlight %}
<div class="col three caption">
CUDA Programming Guide p.22 3.2.2 Device Memory 예제 코드
</div>

line 2-7은 아시다시피 CUDA Kernel 함수입니다. line 15-16에서 host 메모리를, line 20-25에서는 device 메모리를 할당하고 있습니다.
line 27-28은 CUDA Device Memory로 데이터를 복사하고, line 36에서는 CUDA Kernel의 수행 결과를 다시 Host 메모리로 복사하고 있습니다.
끝으로 line 38 이후는 Device 메모리와 Host 메모리를 해제하고 있습니다. 뭔가 복잡해보이지만 사실 간단하죠?

여기서 Device 메모리는 global 메모리라고도 불립니다. 이렇게 중복적인 용어가 등장하는 이유는 CUDA 문서에서 중복해서 사용하기 때문인데, 이제는 On-chip 메모리를 다루는 상황에서 GPU 코어 입장에서는 global하다고 명명하는 것이 한편으론 타당해 보입니다.

또한 앞으로 CUDA 코딩을 하게 되실 분들에게 강력히 권하는 naming이 있습니다. host 메모리는 `h_`로, device 메모리는 `d_`로 시작합니다. 짐작하시겠지만, `h_`는 host 메모리, `d_`는 device 메모리입니다. 이는 포인터 주소같은 것으로는 어디에 할당된 것인지 알 수 없기 때문에, 변수 명으로 구분할 수 밖에 없습니다. 다른 각자의 방법을 고안해서 사용하시거나 회사에서 이미 정해놓은 방법이 있다면 그것을 따르셔도 됩니다만, 기본적으로는 이런식으로 변수명으로 관리를 하시는 것을 권합니다. 또한 CUDA 프로그래밍에서는 다양한 종류의 메모리를 적재적소에 사용하는 것이 중요하므로 다양한 종류의 메모리들이 어떤 메모리에 저장되어 있는지 구별하기 위해서도 이런식의 명명을 따르는 것이 여러모로 유용합니다.

<a id="shared memory" class="anckor"></a>

# Shared 메모리

Shared 메모리는 CUDA On-chip 메모리 중에서 L1캐시와 같은 역할을 합니다. CPU와 가장 큰 차이는, CPU에서는 캐시의 데이터를 스스로 관리를 하지만, CUDA의 L1 캐시라고 불리는 Shared 메모리는 CUDA 코드로 직접 어떤 데이터를 넣을 것인지 명시를 해야만 합니다. 이런 구성이 갖는 장점은 다음과 같습니다.

1. 필요한 만큼의 데이터만을 읽어오게 할 수 있다.
1. global 메모리 접근에 의한 지연시간 감소
1. global 메모리의 대역폭 소모 최적화

또한 shared 메모리는 다음과 같은 특징이 있습니다.

1. CUDA Thread Block 간에서만 데이터 공유가 가능
1. CUDA Computability에 따라 캐시의 크기가 다름
1.


<a id="constant memory" class="anckor"></a>

# Constant 메모리

<a id="texture memory" class="anckor"></a>

# Texture & Surface 메모리
