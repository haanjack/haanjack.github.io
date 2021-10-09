---
layout: post
title:  "Ubuntu에 CUDA 설치"
date:   2016-02-29 0:00:00
publish: false
categories: CUDA
tags:
- CUDA 7.5
- CUDA Driver
---

우분투와 nVidia 그래픽 카드가 설치되어 있는 환경에서 진행한다.

CUDA Toolkit Document에 있는 [Installation Guide Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#axzz41VsznO2m)에 따른다.

### 사전 설치 작업

#### 우분투 버전 확인

CUDA 7.5버전을 위해서는 Ubuntu 14.04가 최소 버전이다. 이는 이 글을 쓰고 있는 현재 LTS버전으로 배포되고 있는 버전이다. 또한 64-bit 버전이어야 한다. 혹시나 어떤 bit를 설치했는지 확인을 하고 싶다면 다음과 같은 명령어를 입력하면 확인할 수 있다. i386이 나오면 x86, amd64가 나온다면 64bit이다.

{% highlight bash %}
dpkg -s libc6 | grep Arch
{% endhighlight %}

#### CUDA 카드 지원확인
다음의 명령을 통해 인식된 카드의 이름 [CUDA 지원 목록](http://developer.nvidia.com/cuda-gpus)에도 나온다면 CUDA 프로그래밍이 가능한 카드이다.
{% highlight bash %}
lspci | grep -i nvidia
{% endhighlight %}

#### GCC 설치 확인
{% highlight bash %}
gcc --version
{% endhighlight %}

나는 build-essential을 처음 설치한 ubuntu를 갖고 CUDA환경을 구성하는 것이어서, build-essential을 설치했다.
{% highlight bash %}
apt-get install build-essential
{% endhighlight %}

#### Kernel Header 및 Development Package 설치 확인
Ubuntu 환경에서 다음과 같은 명령어로 linux header를 설치한다.
{% highlight bash %}
sudo apt-get install linux-headers-$(uname -r)
{% endhighlight %}

#### 설치 방법 선택
CUDA Toolkit을 설치하는 방법은 크게 두가지 방법이 있다.
1. 배포판 의존 패키지
1. Standalone 패키지

Standalone 패키지를 이용하게 되면 여러 리눅스 환경에 쉽게 포팅할 수 있다는 장점이 있지만 업데이트가 사용하는 리눅스 배포판을 따르지 않아 개별적으로 해줘야한다는 단점이 있다. 반면, 배포판 의존 패키지는 패포판 패키지 관리 프로그램으로 업데이트가 관리된다는 장점이 있다. 나는 패포판 의존 패키지를 선택했으며, nVidia 문서에서도 가급적 배포판 의존 패키지를 선택 할 것을 권장하고 있다.

### CUDA Toolkit Download 및 설치 준비
[CUDA 다운로드](http://developer.nvidia.com/cuda-downloads)에서 [Linux - x86_64 - Ubuntu - 14.04 - deb](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb)을 선택했다.

내 경우에는 MAC을 주로 사용하고, 데스크탑을 CUDA 서버로 사용할 계획이었으므로, 주소를 복사해서 다음과 같이 서버로 바로 다운 받았다.
{% highlight bash %}
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb ~/Download
{% endhighlight%}

만약 이전에 설치된 CUDA Toolkit과 Driver가 있다면, 경우에 따라 다음과 같은 명령으로삭제를 해야만 한다. 이때 이전에 설치했던 CUDA Toolkit과 Driver가 runfile로 설치가 되었다면 다음과 같은 명령을 입력한다. X.Y나 package_name 같은건 삭제하게 할때 다시한번 확인해 봐야 할 것 같다.
하지만 만약 runfile을 설치했는데, 다시 runfile로 설치한다거나 deb으로 설치했었는데, 이번에도 deb으로 설치한다면 삭제하는 절차는 하지 않아도 된다고 한다.

{% highlight bash %}
sudo /usr/local/cuda-X.Y/bin/uninstall_cuda_X.Y.pl
sudo /usr/bin/nvidia-uninstall
{% endhighlight %}

한편 Deb으로 설치했다면 다음과 같은 명령을 입력한다.
{% highlight bash %}
sudo apt-get --purge remove <package_name>
{% endhighlight %}

### 설치
설치 명령은 다음과 같다. 다운 받을때 나오는 창에서도 안내로 나온다
{% highlight bash %}
sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb
sudo apt-get update
sudo apt-get install cuda
sudo reboot
{% endhighlight %}

### 관리

#### 설치된 Package 목록 확인
{% highlight bash %}
cat /var/lib/apt/lists/*cuda*Packages | grep "Package:"
{% endhighlight %}

#### 설치한 Package의 업그레이드
패키지 관리자에서 CUDA의 새로운 패키지가 나오면 공지가 나온다고 하는데, 그때 업데이트를 하면 된다고 한다.
{% highlight bash %}
sudo apt-get install cuda
sudo apt-get install cuda-driver
{% endhighlight %}
다만 Ubuntu의 패키지 전체를 업데이트 하는 과정에서 함께 업데이트가 될 것이므로 굳이 별도로 하지 않아도 될 것 같다. 이건 시간이 지나서 실제 업데이트가 될 때 어떻게 되는지 지켜봐야 할 것 같다.


### 시험
CUDA Toolkit 및 driver 설치와 함께 설치된 sample을 통해서 CUDA가 올바르게 설치가 되었는지 확인해 보자.

#### 환경변수 추가
사용하고 있는 쉘에 다음과 같은 라인 두개를 추가한다.
{% highlight bash %}
export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
{% endhighlight %}

Sample을 home에 복사
{% highlight bash %}
cuda-install-samples-7.5.sh ~
cd ~
cd NVIDIA_CUDA-7.5_Samples
make
{% endhighlight %}

Compile이 끝나면 다음과 같이 Sample을 실행해보자. 캡쳐한 그림과 비슷한 화면이 나오면 성공적으로 설치가 된 것이다. 또한 화면에 나온 내용을 통해서, 설치된 CUDA 카드의 스펙을 확인할 수 있다.
{% highlight bash %}
./NVIDIA_CUDA-7.5_Samples/bin/x86_64/linux/release/deviceQuery
{% endhighlight%}

<img class="col" src="{{site.info.baseurl}}/images//cuda_deviceQuery.png"/>

{% highlight bash %}
./NVIDIA_CUDA-7.5_Samples/bin/x86_64/linux/release/simpleOccupacy
{% endhighlight %}

<img class="col" src="{{site.info.baseurl}}/images//cuda_simpleOccupancey.png"/>
