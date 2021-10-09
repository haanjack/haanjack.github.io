---
layout: post
title:  "GPGPU와 CUDA"
date:   2016-02-26 2:18:00
description: GPGPU를 위한 CUDA에 대한 배경 지식. 그리고 GPU 구매 가이드
publish: false
categories: CUDA
tags:
- CUDA Architecture
- GPGPU
---

### CPU vs GPU, 그리고 GPGPU
#### CPU와 GPU의 성능 비교
일반적으로 Computer에서 연산을 처리하는 모듈은 프로그램의 실행 및 시스템 자원의 관리 등에 특화 되어 있는 CPU와 영상처리를 위한 GPU로 구성되어 있습니다. 이중에서 GPU라는 것은 흔히 그래픽 카드라고 불리는 확장카드에 있는 코어를 말합니다. 과거에는 별도의 카드를 통해서 설치를 해야 했지만, 요즘은 칩 접적기술이 발달하게 되면서 대부분의 컴퓨터에서는 Mainboard에 함께 집적된 상태로 또는 CPU와 같은 칩셋에 집적된 상태로 판매되고 있습니다.

![상황이 이렇다보니 2014년 당시 그래픽 칩셋의 점유율 자체는 인텔이 가장 높습니다.]({{site.info.baseurl}}/images//insidelogo.jpg)

하지만 이러한 그래픽 칩셋들은 어디까지나 일반적인 사용환경(오피스, 웹서핑, 영화 등)에 초점을 맞춘 그래픽 카드를 포함한 것이고, 게임을 원활이 즐기는데 충분한 그래픽카드라고는 할 수는 없습니다. 대신 확장 그래픽 카드, 그러니까 지금도 계속해서 PC에 확장카드 형태로 설치하는 그래픽 카드는 앞서 말한 온보드 내지는 온칩 형태의 그래픽 칩셋보다는 웬만해서는 좋은 그래픽스 연산 성능을 갖고 있습니다.

그렇다면 GPU의 연산 성능은 얼마나 CPU보다 좋을까요.

![Intel CPU와 nVidia GPU간의 성능 비교]({{site.info.baseurl}}/images//CPUvsGPU.png)

GPU의 성능은 2000년대 중반 이후부터는 CPU보다도 월등히 높은 성능을 보여주고 있는데, 이는 CPU가 구성되는 특성상 CPU가 가질수 밖에 없는 제약조건 때문입니다. 과거에는 Clock Speed를 높이고 집적도를 늘리면 성능이 증가했지만 인텔 혼자서는 더 이상의 성능을 내기 힘든 상태를 만나게 된 것입니다.

![Sandia의  Multicore simulation 결과]({{site.info.baseurl}}/images//many_core_bw_sandia.jpg)

위 자료는 Sandia에서 Multicore CPU를 시뮬레이션을 했을때 메모리가 시스템 성능에 미치는 영향을 보여줍니다. 즉, 인텔이 아무리 집적기술이 좋다고 하더라도 모두가 사용하는 PC환경에서는 CORE가 많은 것이 최선이 아니라는 것을 의미하는 것이죠.

하지만 GPU는 CPU와 다른 길을 갈 수 있습니다. 시스템의 자원들을 관리할 필요도 없다보니 데이터를 병렬적으로 처리하는 본연의 동작에 집중할 수 있고, 위 그래프에서와 같이 Core에 Memory를 올리는 것 처럼 GPU의 동작 조건을 구성할 수 있는 것이죠.

하지만 이렇게 성능을 높일 수 있다고 하더라도 제조사 입장에서는 고성능을 위해 비싸진 칩셋을 어떻게 팔아야할지 고민을 안할 수 없습니다. 또한 컴퓨터를 활용하여 실험이나 시뮬레이션을 해왔던 여러 연구원들의 입장에서는 CPU를 더 사고, Cluster를 구성해야만 하는 것에서 보다 저렴하면서도 좋은 성능을 낼 수 있는 수단이 필요했습니다.

#### GPGPU의 등장
GPGPU (General Purpose Graphic Processing Unit)이란 개념은 점차 발전해가고 있는 GPU의 Computing 성능을 지금까지 사용해온 그래픽스 목적 외에 보다 범용적인 용도로 사용할 수 있도록 하자는 데서 출발합니다. CPU에서 SIMD(Single Instruction Multiple Data) 인스트럭션으로 연산 성능을 높이고 Assembly등을 통해서 연산 성능상의 제약을 극복하고자 했던 노력에서 GPU를 연산 가속기로 이용할 수 있게 된 것입니다.

![CPU에서 지원하는 SIMD 인스트럭션 목록]({{site.info.baseurl}}/images//220px-X86_extensions_2013.svg.png)

물론 그 당시에도 그래픽스 프로그래밍은 할 수 있었습니다. 하지만 OpenGL, DirectX를 공부하고, 쉐이더 언어(HLSL, GLSL, Cg 등)를 배워야 했으며, 그래픽스를 목적으로 만들어진 개발환경이라 아무래도 공부할 것도 많고 제약사항도 많았습니다. 이 글을 쓰고 있는 저 역시 저 언어들은 할 줄 모릅니다.

그 대신 GPGPU에 적합한 개발 환경이 등장하게 되면서, GPU프로그래밍을 그래픽스 분야 전문가 뿐만 아니라 C/C++을 할 수만 있다면 여러 연구분야에서 가져다 쓸 수 있도록 진입장벽이 낮아지게 됩니다. nVidia의 CUDA를 필두로 하여, [OpenCL](https://www.khronos.org/opencl/), DirectCompute 등이 그 주인공입니다.

![]({{site.info.baseurl}}/images//NV_DesignedFor_CUDA_3D_sm.png)
![]({{site.info.baseurl}}/images//OpenCL_Logo.png)
![]({{site.info.baseurl}}/images//directx-11-logo.png)

CUDA는 nVidia에서 개발한 언어라는 점에서 알 수 있듯이 nVidia 그래픽카드에서만 동작합니다. 반면 OpenCL는 OpenGL등을 발표하는 [KronosGroup](https://www.khronos.org)에서 여러 회사에서 모인 표준안인지라 병렬지원을 하는 칩셋이라면 지원을 합니다. Apple에서 주도한 것으로 알려져 있고, 대표적으로 Intel, AMD의 CPU와 nVidia의 GPU와 CPU에서 동작합니다. DirectCompute는 DirextX11과 함께 출시가 되었습니다.

#### 무엇을 공부할 것인가
그렇다면 어떤 것을 공부하면 좋을까요.
저는 개인적으로 CUDA를 공부하는 것을 추천합니다. nVidia 그래픽카드만 되는 것은 분명한 한계입니다. 하지만 Breakpoint가 가능한 디버거를 제공하고, 프로파일러, Best Practice Guide 등을 제공합니다. 따라서 쉽게 개발할 수 있고, 보유하고 있는 GPU자원을 최대한 끌어다 쓸 수가 있습니다.

하지만 다른 두가지 언어는 범용적이란 장점을 제외하면 개발이 수월하지 않습니다. 디버깅이 가능하지 않고, 여러 시스템을 지원해야하므로 최적화가 불가능합니다. 심지어는 제가 해보았을떄는 Best Practice로 개발한 Image Convolution CUDA 커널을 CUDA와 OpenCL로 옮기고 해보아도 OpenCL이 더 느렸습니다. 더욱이 Intel, AMD는 CPU에서만 지원합니다. 고성능을 목적으로 공부해서 사용하려는데 CPU에서 동작시킬 것이라면 저라면 차라리 OpenMP를 사용하겠습니다.

만약 부득이하게 OpenCL로 개발을 하셔야 한다면 CUDA로 개발을 한 이후에, OpenCL로 번역하실 것을 권합니다. nVidia가 CUDA개발을 해둔 상태에서 참여하게 되어서 많은 부분이 명칭만 다를 뿐 비슷한 개념을 가진 키워드들이 있습니다. 따라서 디버깅이 되지 않는 환경에서 굳이 애써 개발하시기 보다는 개발이 쉬운 CUDA에서 개발 한 이후에 진행하는 것이 쉽다는 것이 여러모로 유리합니다.

#### 어떤 GPU를 살까
nVidia의 그래픽카드의 제품군은 GeForce, Quadro, Tesla 등이 꾸준히 유지되고 있지만, 실제 그 안에서의 기술은 빠르게 변하고 있습니다. 전 이 도표를 2010년에 처음보고, nVidia가 과연 할 수 있을까 의심을 했었지만, 지금까지 nVidia는 자신의 약속을 지켜오고 있습니다.
![]({{site.info.baseurl}}/images//NVIDIA-2016-Roadmap-Pascal-GPU.jpg)

[CUDA Capability](https://en.wikipedia.org/wiki/CUDA#Supported_GPUs)를 보면 CUDA Capability와 CUDA Core의 버전을 분류해놓은 테이블이 나옵니다. 기능은 버전이 올라갈 수록 아래 그림처럼 기능이 계속해서 추가 되어 왔습니다. 이런 부분은 CUDA 프로그래밍을 하면서 편리해지고 성능을 높이는데 많은 도움이 됩니다. 이 그림 말고도 참고해야할 것들이 많습니다만, 그건 본격적인 CUDA 프로그래밍을 할때 참고하는 것이 좋겠습니다.

![]({{site.info.baseurl}}/images//tech_specs.jpg)

중요한 것은, Goole에서 공개한 tensorflow는 CUDA capability 3.5버전을 최소 사양으로 정해놓았습니다. 제가 사용하는 GPU는 GeForce GTX 660이고 Core 세대는 Kepler인데, CUDA Capability는 3.0입니다. 다행인 것은 github에 임의로 3.0을 지원할 수 있도록 하는 [Tip](https://github.com/tensorflow/tensorflow/issues/25)이 있습니다.

아무튼 새로 사실 예정이라면, CUDA Capability 3.5이상인 GPU를 구매하시는 것이 좋으며, 목록은 [CUDA Capability](https://en.wikipedia.org/wiki/CUDA#Supported_GPUs)에서 예산에 맞는 가급적 가장 높은 Capability를 지원하는 카드를 사시는 것이 좋겠습니다.
