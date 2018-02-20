---
layout: post
title:  "docker 명령 만으로 nvidia-docker 사용하는 방법"
description: "6 글자를 치는 것이 너무 힘들어서 쓰는 글"
date:   2018-02-21
published: true
categories: docker
tags:
 - docker
 - nvidia-docker
 - NGC
 - NVIDIA
---

*```nvidia-docker```를 이용하면 매번 ```nvidia-docker```라고 쳐야 하나요?*

사실 시작할때 한번만 nvidia-docker를 치면 되는 문제이지만, 사실 이 문제는 그리 간단하지 않다.
나 스스로도 profile에 **alias docker=nvidia-docker**를 넣고 지냈던 적이 있었기에 별거 아니지만 신경이 쓰이는 부분이다.
더욱이 nvidia-docker2에 이르러서는 nvidia-docker를 이용하는 방법이 하나 더 추가되어서 두가지 방법을 이용할 수 있다.

{% highlight bash %}
$ nvidia-docker run ... #기존 방법
$ docker run --runtime=nvidia ... #새로 추가된 방법
{% endhighlight %}

새로운 방법이 가능해지면서, nvidia-docker를 지원하지 않던 orchestration 도구들에 대해서 시험해 볼 수 있는 기회가 열렸다.
그러나 여전히 추가로 쳐야만 한다는 어려움(?)이 남아 있다. 게다가 pycharm에서는 runtime을 docker remote interpreter 옵션에서 지원하지 않으므로, 어쨌건 native하게 사용할 수가 없다.

그러면 어떻게 이를 해결할 수 있을까?

# (Option 1) alias 설정
.bashrc나 .zshrc에서 다음 라인을 추가한다.
{% highlight bash %}
alias docker=nvidia-docker
{% endhighlight %}

이후 프로파일을 다시 로딩하면 잘 된다. 사실 interactive하게만 사용한다면 이 방법만으로도 충분하다.

# (Option 2) docker daemon.json에 default-runtime 추가

/etc/docker/daemon.json을 다음과 같이 수정한다. 추가해야할 라인은 두번째 줄에 있는 것이다.

{% highlight bash %}
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
{% endhighlight %}

이렇게 하면 nvidia-docker라고도 --runtime=nvidia 옵션도 칠 필요 없이 바로 실행을 할 수 있고 docker-compose도 이용할 수 있다.

<img class="col three center" src="/images/201801/docker.png">

이 방법이 위 방법보다 갖는 장점은 docker-compose도 지원할 수 있다는 것이다.

가령 아래와 같이 docker-compose.yml 파일이 있다고 하자.
{% highlight bash %}
version: '3'

services:
  cuda:
    image: nvidia/cuda:9.1-base-ubuntu16.04
    command:
      - nvidia-smi
{% endhighlight %}

이를 간단히 아래의 명령으로 실행이 가능하다.

<img class="col three center" src="/images/201801/docker-compose.png"/>

그렇다면 이것으로 docker orchestration을 할 수 있는 길이 열렸다고 볼 수 있겠다.
당장 나는 [pycharm](https://www.jetbrains.com/help/pycharm/docker-using-docker-as-a-remote-interpreter.html)을 붙였는데, [Kubernetes](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)도 붙여볼 생각이다.

덧,
pycharm에 붙여서 사용하는 방법은 현실적으로 local과 remote를 구분해서 쓰는 것이 맞는 것 같기도 하다. ([참고글](http://killianlevacher.github.io/blog/posts/post-2016-07-22/post.html))
<br>
