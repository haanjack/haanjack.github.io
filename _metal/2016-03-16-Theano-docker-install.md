---
layout: post
title:  "Debian docker에 Theano 설치하기"
date:   2016-03-16 00:00:00
categories: Machine Learning
description: nVidia docker 이미지 기반 Python/Theano 개발 환경 구축 (GPU/CPU)
published: false
tags:
- Theano
- CUDA
- GPU
- docker
- debian
---

## Theano

Theano는 Machine Learning에 필요한 알고리즘을 제공하는 Python Library이다. University fo Montreal에서 주도적으로 개발 하고 있으며, 라이브러리에서 제공하는 알고리즘의 상세한 설명은 Yoshua Bengio 교수가 쓰고 있는 책에서 볼 수 있다. 코드가 오픈되어 있고 GPU 프로그래밍까지 다룰 수 있고 문서도 잘 되어 있어서, 좋은 공부용 라이브러리라고 할 수 있다. 다만, 머신러닝 스터디 중간에 들아가다보니 내용을 쉽게 따라 잡고 있지 못하고는 있는데, 해당 내용에 대해서는 Theano의 Deep Learning Tutorial을 우선 해보고 진입할 생각이다.

## 개발 환경 준비

우선 Theano를 해보기 위해서는 Theano를 개발환경에 설치해야 한다. 우선 나는 windows를 은행을 위해서만 사용하므로 개발환경에 대한 논의에서 제외한다.

나는 기존에 Dockerfile로 개발환경을 준비해 두었고, 이글을 보시는 분들도 하나의 설치 사례로 참고해 두어도 좋을 것이라고 생각한다. 다만 이런 모든 과정이 귀찮고 바로 넘어가고자 하는 경우에는 다양한 Dockerfile을 가진 Kaixhin의 Github 저장소에서 필요한 Dockerfile을 가져오면 된다.

## GPU(CUDA)를 사용하는 경우

> 필요한것
>
> CUDA 개발환경이 구축이 된 Linux 환경 (설치방법은 별도의 포스트에서 참조)
> CUDA를 사용할 수 있는 Dockerfile 및 docker가 설치된 linux 환경

여기서는 CUDA 및 docker를 사용할 수 있는 linux 환경이 준비되어 있다는 전제하에 다음 내용을 진행한다.

1. CUDA Enabled Dockerfile 준비

nVidia에서 제공하는 CUDA enabled된 이미지를 사용한다. 그렇게 하는 경우, 아래 그림과 같이 여러 이미지에서 GPU를 가상화하여 사용할 수 있다.

<a href="https://github.com/NVIDIA/nvidia-docker"><img class='col two center' src='https://cloud.githubusercontent.com/assets/3028125/12213714/5b208976-b632-11e5-8406-38d379ec46aa.png'/></a>

우분투를 베이스로 이용한다면 다음과 같은 절차를 따르면 된다.

{% highlight shell %}
git clone https://github.com/NVIDIA/nvidia-docker.git
cd nvidia-docker

# Initial setup
sudo make install
sudo nvidia-docker volume setup

# Run nvidia-smi
nvidia-docker run nvidia/cuda nvidia-smi
{% endhighlight %}

하지만 나는 Debian Image를 베이스로 사용하므로 clone 한 저장소의 디렉토리 안에 있는 Makefile의 내용을 일부 수정했다.
{% highlight shell %}
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.

NV_DOCKER ?= docker

OS ?= ubuntu-14.04
USERNAME ?= nvidia
WITH_PUSH_SUFFIX ?= 0
ifeq ($(WITH_PUSH_SUFFIX), 1)
  PUSH_SUFFIX := -$(subst -,,$(OS))
endif

(...)
{% endhighlight %}

이것을 다음과 같이 바꾸었다. (5,6번 줄 참조)

{% highlight shell %}
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.

NV_DOCKER ?= docker

OS ?= debian
USERNAME ?= hanjack
WITH_PUSH_SUFFIX ?= 0
ifeq ($(WITH_PUSH_SUFFIX), 1)
  PUSH_SUFFIX := -$(subst -,,$(OS))
endif

(...)
{% endhighlight %}

사실 6번째 줄의 username이 어떤 의미가 있는지 잘 모르겠다.
docker에 사용자를 추가하려면 useradd나 adduser를 사용해야 한다.

여튼 위와 같이 수정을 하고, 앞서 Ubuntu에서 nvidia-docker 이미지를 설치했을때 사용하는 명령어(line 5이후)를 사용한다. 그러면 최종적으로 docker에 설치된 이미지를 확인할 수 있는데, 다음 그림과 같다.

[그림 넣어라]

2. Python 개발 환경 설치 Docker로 개발 환경이 추가된 이미지 생성

1절에서 만든 이미지에 앞으로 개발하는데 사용할 Python이나 jupyter 등을 별도의 Dockerfile을 이용하여 이미지를 빌드하였다.

{% highlight shell %}
docker build -t metal/cuda . # 마지막 점은 Dockerfile이 있는 경로
{% endhighlight %}

여기서 -t 태그 옵션으로 이미지의 이름을 지정해주는 역할을 한다. 사용하는 각자가 원하는 이름을 사용하면 된다.

**개발환경을 설치하기 위한 Dockerfile**
<a id="python-dev-install"></a>

{% highlight shell %}
FROM nvdocker-build
MAINTAINER hanjack

LABEL Description="This image is the base of python and octave app" Vendor="LucidMind Ltd." Version="1.0"

USER root

# Debian Update
RUN apt-get update && apt-get -y upgrade

# install Aptitude
RUN apt-get update && apt-get -y install --no-install-recommends\
  aptitude

# Install Python 2
RUN aptitude update && aptitude -y --without-recommends install\
  zsh git vim curl wget\
  build-essential\
  python python-dev python-pip\
  python-matplotlib\
  python-numpy python-scipy\
  python-pandas\
  python-sympy python-nose\
  libhdf5-dev\
  python-h5py\
  python-tables\
  python-zmq\
  libzmq3-dev\
  python-tk\
  python-pexpect\
  python-simplegeneric\
  cython\
  && aptitude clean

# Install jupyter & kernel spec globally to avoid permission problems when NB_UID
# switching at runtime.
RUN pip install\
  jupyter
RUN jupyter kernelspec install-self
EXPOSE 8888

# Setup jupyter
RUN echo "jupyter notebook --ip=0.0.0.0 --no-browser --notebook-dir='/workspace'" > /run_notebook.sh &&\
    chmod +x /run_notebook.sh &&\
    mkdir -p `jupyter --config-dir`/custom &&\
    wget https://raw.githubusercontent.com/haanjack/customfiles/master/jupyter/ &&\custom.css -O `jupyter --config-dir`/custom/custom.css

# Install Octave
RUN aptitude update && aptitude -y  --without-recommends install\
  octave\
  && aptitude clean
RUN pip install octave_kernel
RUN python -m octave_kernel.install

# Create ipython profile & Configuration
RUN ipython profile create notebook

# Installation for zsh & Oh-my-zsh
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

# Working Information
WORKDIR /workspace

CMD ["/bin/zsh"]
{% endhighlight %}

이후 Theano를 설치하기 위한 Dockerfile을 만들어, 앞서 만든 이미지에 확장하여 Theano가 설치된 이미지아 그렇지 않은 이미지를 구분할 수 있도록 하였다.

다만, 위 Dockerfile에는 Octave와 Jupyter의 Thema를 설치하므로 원하지 않는다면 삭제하고 사용하여야 한다.

다음 theano를 설치하기 위한 Dockerfile을 준비한다.
<a id="theano-install"> </a>

{% highlight shell %}
# Copyright (c) hanjack's LucidMind Project
# Distributed under the terms of the Modified BSD License.

FROM metal/cuda
MAINTAINER hanjack <haanjack@gmail.com>

LABEL Description="This image is the base of python and octave app" Vendor="LucidMind Ltd." Version="1.0"

USER root

# install theano
RUN aptitude install liblapack-dev libblas-dev gfortran --without-recommends -y &&\
    pip install --upgrade git+git://github.com/Theano/Theano.git &&\
    theano-nose
{% endhighlight %}

파일을 준비한 후 다음 명령을 입력하면 기존의 이미지를 참조하여 theano에 필요한 추가항목들을 설치한 새로운 이미지가 생성된다.

{% highlight shell %}
$ docker build -t metal/cuda/theano .
{% endhighlight %}


그러면 meta/cuda/theano와 같이 사용자가 원하는 이미지 명을 가진 docker image가 생성 될 것이다.

### CPU만을 사용하는 경우

> 필요한 것
>
> docker가 설치된 mac 또는 linux 환경

앞에 GPU환경에서 사용했던, Python 개발환경을 설치하기 위한 <a data-scroll href="#python-dev-install">Dockerfile</a>에서, from라인을 debian/latest로 수정한다. 그러면 필요한 개발환경이 설치된 debian 이미지를 얻을 수 있다. 여기에 Theano를 설치하기 위한 <a data-scroll href="#theano-install">Dockerfile</a>을 추가로 적용하여, theano가 설치된 이미지를 만들 수 있다.

물론 4번째 라인의 FROM에 들어갈 내용은 앞서, 개발환경을 설치하면서 사용한 이미지 이름을 지정해주어야 한다.
