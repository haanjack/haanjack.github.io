---
layout: post
title:  "nvidia-docker를 이용해서 GPU 나눠 쓰기"
description: "여럿이 나눠써야 하거나, 작업에 따라 구분해서 써야 할 경우"
date:   2018-04-14
published: true
categories: docker
tags:
 - docker
 - nvidia-docker
 - NGC
 - NVIDIA
---

[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)를 이용하여 docker 내에서 GPU를 이용할 수 있는 것은 익히 알려진 사실이다.

한편 nvidia-docker를 갖고 할 수 있는 것 중 하나는 복수의 GPU가 설치된 환경에서 GPU를 나눠서 사용할 수 있다는 것이다. 이 내용은 [nvidia-container-runtime](https://github.com/nvidia/nvidia-container-runtime)의 매뉴얼에 간단히 정리가 되어 있는 내용이며, NVIDIA_VISIBLE_DEVICES의 항목을 참고해도 된다.

GPU를 나눠 쓰는데 중요한 것은 내가 나눠 쓰고자 하는 GPU의 index 순서와 PCI bus의 주소가 일치하여 잘라지는 것이냐는 것이다. 이것이 중요한 이유는 [NUMA Architecture](https://ko.wikipedia.org/wiki/%EB%B6%88%EA%B7%A0%EC%9D%BC_%EA%B8%B0%EC%96%B5_%EC%9E%A5%EC%B9%98_%EC%A0%91%EA%B7%BC)에서 [PCI root complex](https://en.wikipedia.org/wiki/Root_complex)를 고려하여 GPU를 나눠 쓸 때 의도한 대로 GPU를 나눠 쓸 수 있는지 여부를 결정하는 요인이 되기 때문이다.

우선 내가 가진 GPU ~~서버~~ 워크스테이션에서 보이는 GPU의 정보는 다음과 같다.
<img class="col two center" src="/images/2018/nvidia-smi.png"/>

# nvidia-docker2에서 GPU를 나눠서 활용하는 방법

nvidia-docker2에서 나눠 쓰는 방법은 **NVIDIA_VISIBLE_DEVICES** 를 이용하는 것이다. 예를 들면 다음과 같은 명령을 주면 0번 GPU만을 선택할 수 있다.
{% highlight bash %}
$ docker run --rm -ti --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 nvidia/cuda nvidia-smi
{% endhighlight %}

복수로 지정을 해야하는 경우, 간단히 다음과 같이 할 수도 있다.
{% highlight bash %}
$ docker run --rm -ti --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1,2 nvidia/cuda nvidia-smi
{% endhighlight %}
<img class="col two center" src="/images/2018/nvidia-docker-isolation.png"/>

nvidia-smi에 나오는 Bus-Id를 확인해 보면, 1,2번 GPU가 0,1으로 잘 들어오는 것을 확인할 수 있다. 이런 방법을 이용해서 GPU application에서 GPU의 물리적인 index를 고려하지 않고 개발 및 운용을 하도록 할 수 있다.

# nvidia-docker에서 GPU를 나누는 방법

사실 nvidia-docker의 예전 버전에서는 다른 방법으로 GPU isolation을 지원해 왔었다. 이를 테면 다음과 같이 명령을 할 수 있다.
{% highlight bash %}
$ NV_GPU=1,2 nvidia-docker run --rm -ti nvidia/cuda nvidia-smi
{% endhighlight %}

실행 결과는 위와 동일하게 나온다.

<img class="col two center" src="/images/2018/nv-gpu-isolation.png"/>
