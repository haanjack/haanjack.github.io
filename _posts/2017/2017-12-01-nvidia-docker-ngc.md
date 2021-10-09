---
layout: post
title:  "Docker를 활용한 deep learning 개발 환경 구축"
description: "Ubuntu 환경에서 docker를 이용하여 10분 안에 deep learning 시작하기"
date:   2017-12-01
published: false
categories: docker
tags:
 - docker
 - nvidia-docker
 - NGC
 - NVIDIA
---

*일전에 작성하였던 [nvidia-docker를 활용한 deep learning 환경 구축]()이란 글이 있지만,
나의 사용방법도 변화하였고 nvidia-docker 2.0이 나왔기에 새로 작성하게 되었다. (이전 글은 내려놨다...)*

# Introduction
일단은 docker를 이용하는지에 대해선 이 글을 보시는 분들은 docker의 장점을 이미 알고 계실 것이라 생각하고 생각하려 한다. 아니면 docker를 쓰면 좋다는 이야기를 듣고 알아보고자 하시는 것이리라.

이 글은 간단한 Tutorial을 통해서 GPU가 구동되는 Deep Learning Framework을 구동하는 것이 이 글의 목표이다.
이후에 어떤 Customizing을 할 수 있는지는 별도의 글을 통해서 제공할 것이다. 다만 어떤 방향으로 이를 활용할 수 있는지는 글의 끝머리에 간단히 이야기해보려 한다.

여기서는 *nvidia-docker*와 *NGC*를 설명할 것인데, 이는 내가 생각하기에 바로 가져다 쓰는 용도로는 가장 최적의 조합이라고 생각하기 때문이다.
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)는 docker container 내에서 NVIDIA GPU를 이용할 수 있도록 해주는 plug-in 또는 Wrapper이다. 따라서  즉, 독립적(standalone)으로 동작하지 못하므로, docker를 먼저 설치해야만 한다. [NGC](https://ngc.nvidia.com/signin/email)는 NVIDIA에서 공개한 Docker registry로서 [NVIDIA Dockerhub](https://hub.docker.com/u/nvidia/)을 통해 제공하던 것을 보다 전략적으로 배포하기 위해서 만든 서비스라고 볼 수 있다. NGC에 대해서는 별도의 글을 통해서 설명을 할 것이다.

# 이용환경

원활한 실습을 위해 다음의 환경이 갖춰져 있어야 한다.

1. 리눅스가 설치된 Desktop 또는 서버
  - Ubuntu나 CentOS 7이라면 문제 없음. Kernel이 3.X 대인 CentOS 6은 docker의 성능에 문제가 있다고 한다.
    여기서는 Ubuntu 환경을 기준으로 설명을 하겠지만, CentOS를 전에 이용했을때도 원활히 이용할 수 있었다.
  - Windows에서는 docker가 VM위에서 동작하며, VM은 NVIDIA GPU를 지원하지 않으므로 제외
  - MAC은 될지도 모르겠으나 내가 해볼 수 없으므로 이런 환경을 이용하시는 분은 자체적으로 해결하시거나 리눅스를 이용하시라고 말씀드려야 할 것 같다.
2. NVIDIA GPU
  - NVIDIA GPU가 있어야만 nvidia-docker를 설치할 수 있고, container에서 이용할 수 있다.
  - GPU의 종류는 CUDA를 이용할 수 있는 모든 NVIDIA GPU라면 상관 없다. 다만 Jetson 이용자라면 아직 안해봐서 모르겠다.

# 설치

완전히 새롭게 리눅스를 설치한 환경에서 다음의 절차 만으로 container 내에서 NVIDIA GPU를 이용할 수 있다.

1. NVIDIA Driver 설치

{% highlight bash %}
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-387
sudo reboot
{% endhighlight %}

2. Docker Install

docker의 무료버전인 docker-ce를 설치한다.
{% highlight bash %}
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
{% endhighlight %}

Ubuntu 환경이 아닌 분들은 [docker 공식 설치 안내](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#set-up-the-repository)을 참고해서 사용하는 배포판에 맞는 버전을 설치하면 된다.

다음 docker를 설치해주면 되는데, **nvidia-docker를 이용할 것이라면 굳이 하지 않아도 된다**. 이는 nvidia-docker에서 의존성을 확인하므로 알아서 설치를 해주기 때문이다. 그래도 설치를 하고 싶다면 아래와 같이 입력하면 된다.

{% highlight bash %}
sudo apt-get install docker-ce
{% endhighlight %}

그리고 추가로 docker를 사용할 때 항상 ```sudo```를 입력해야하는데, 이것이 번거로워서 매번 sudo를 입력하지 않아도 되도록 하기 위해서는 다음을 입력한다.

{% highlight bash %}
sudo usermod -aG docker $USER
{% endhighlight %}

3. nvidia-docker Install

nvidia-docker의 버전이 2.0이 되면서 설치 방법에 변화가 생겼는데,
이전에는 deb이나 rpm 같은 패키지를 받아서 처리하는 방식이었다면 이제는 repository를 추가하여 처리하는 것으로 변경되었다.

{% highlight bash %}
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
{% endhighlight %}

여기서도 역시 다른 배포판을 이용한다면 [nvidia-docker 설치 안내문서](https://nvidia.github.io/nvidia-docker/)를 참고하면 된다. 기존 nvidia-docker는 패키지를 제공하는 방식이었는데, 저장소를 이용하게 되면서 얻는 장점으로는 이전에는 docker가 update되는 과정에서 nvidia-docker가 지원하지 않으면서 조금전만 해도 되던 docker가 안되는 상황을 방지할 수 있게 되었다는 것이다.

한편 주의할 점은 만약 이전에 nvidia-docker를 이용하고 있었다면, 사전에 삭제를 먼저 해야만 한다.

**Ubuntu distributions**
{% highlight bash %}
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge nvidia-docker
{% endhighlight %}

**CentOS distributions**
{% highlight bash %}
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo yum remove nvidia-docker
{% endhighlight %}

# docker 동작 확인

이제 모든 준비가 되었다. 다음의 명령으로 Container에서 nvidia-gpu가 정상적으로 인식이 되는지 확인할 수 있다.

{% highlight bash %}
docker pull nvidia/cuda
docker run --runtime=nvidia --rm -ti nvidia/cuda nvidia-smi
{% endhighlight %}

<img class="col two center" src="/images/201712/docker-nvidia-smi.png"/>

한편 이전의 명령도 함께 지원해준다.
{% highlight bash %}
nvidia-docker run --rm -ti nvidia/cuda nvidia-smi
{% endhighlight %}

<img class="col two center" src="/images/201712/nvidia-docker-smi.png"/>

무엇이 편할지는 각자가 판단할 부분이라고 생각한다. docker를 쓰다보면 각종 파라미터 입력이 귀찮아서 shell을 만들기도 하는데, 어떤것이 편할지는 역시 각자가 판단할 부분이다.

# GPU 나눠 쓰기
nvidia-docker는 한가지 더 유용한 기능을 갖고 있는데 다음과 같이 입력하면 GPU를 나눠 쓸 수 있다. GPU가 4개가 있는 시스템을 예로 들어보겠다.
{% highlight bash %}
nvidia-docker run --rm -ti nvidia/cuda nvidia-smi
{% endhighlight %}

<img class="col two center" src="/images/201712/nvidia-docker-4gpu.png"/>

이는 과거 명령인 ```nvidia-docker```에서도 동일하게 동작한다.
{% highlight bash %}
NV_GPU=0,1 nvidia-docker run --rm -ti nvidia/cuda nvidia-smi
{% endhighlight %}

<img class="col two center" src="/images/201712/nvidia-docker-2gpu.png"/>

# NVIDIA GPU CLOUD
2017년 10월 26일, NVIDIA는 NGC라는 것을 발표한다. 그 그림은 여러 사람을 혼란에 빠지게 했었는데, 마치 NVIDIA가 AWS나 GCP와 같이 Cloud 사업에 진출하는 것처럼 보였기 때문이다.
<img class="col two center" src="/images/201712/NVIDIA GPU Cloud for Deep Learning, HPC Applications and HPC Visualization.png"/>
하지만 그것이 아니고 registry로서, nvidia에서 제공한 container를 기반으로 개발하고, cloud를 통해서 배포하라는 의미로 이름을 그렇게 지은 것이라고 한다.

NGC를 이용하는 방법은 *NGC에 가입 및 활용하는 방법 (작성중)*을 확인해서 사용하면 된다. NGC는 https://ngc.nvidia.com 에서 가입 및 계정을 이용할 수 있으며, 웹페이지에도 설명이 있으므로 참고해서 이용할 수 있다. 웹페이지에 접속하면 다음과 같은 창이 나온다.
<img class="col two center" src="/images/201712/ngc-intro.png"/>

이를 docker에서 다음과 같이 가져올 수 있다.

{% highlight bash %}
docker pull nvcr.io/nvidia/tensorflow:17.12
docker pull nvcr.io/nvidia/caffe:17.12
{% endhighlight %}

즉, framework 별로 이름이 달리 되어 관리되고 있고, 버전은 tag 항목에서 17.12와 같이 "년도.월"로 관리가 된다. 만일, TF같이 한달이 멀다하고 버전이 올라가는 Framework에서는 다음과 같이 container에게 물어보고 버전을 알 수 있다.

<img class="col center" src="/images/201712/ngc-tf-version.png"/>

이 Container에는 NVIDIA에서 준비한 예제 코드들이 들어가져 있어서 바로 시험을 해볼 수 있다. ()테스트 조건은 Resnet50 이용, Batch Size는 64, GPU는 4개 사용)

{% highlight bash %}
nvidia-docker run --rm -ti nvcr.io/nvidia/tensorflow:17.12 \
    ./nvidia-examples/cnn/nvcnn.py -m resnet50 -b 64 -g 4
{% endhighlight %}

그러면 아래와 같은 화면이 나온다.
<img class="col two center" src="/images/201712/tf-test-start.png"/>
<p class="center">중략</p>
<img class="col two center" src="/images/201712/tf-test-end.png"/>

그러면 Tesla P100 GPU 4장을 사용한 환경에서 synthetic image를 활용하여 초당 820장 정도의 학습이 된다는 것을 알 수 있게 된다. tfrecord를 만들어 활용하더라도 model이 GPU에서 소비하는 데이터 량이 Disk I/O보다 떨어지므로 그리 큰 영향을 미치지는 않는다.
이제 각자 시험해보고 싶은 모델과 데이터를 갖고 시험을 해볼 수 있는 환경이 구성이 되었다.

<br/>
