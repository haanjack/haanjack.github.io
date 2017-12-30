---
layout: post
title:  "Theano 시작하기"
date:   2016-02-29 0:30:00
categories: CUDA Python Theano
published: false
---

# Theano 시작하기

## 설치
언제나 그렇듯 설치는 무엇을 배우는데 있어서 쉬운 편이다.

#### 사전에 설치해야 하는 것
python, python, python-dev, NumPy, SciPy, nose, sphinx, pip, pydot

나는 [conda]()를 설치해서 한번에 끝냈다. 가급적 직접 설치해서 쓰고 싶은데, 기본적으로 깔린 python2.7을 어찌할수가 없어서 이 편이 훨씬 건강에 좋은것 같다.

### 설치 명령
{% highlight shell %}
pip install Theano
{% endhighlight%}

소스도 받고 싶다면,
{% highlight shell %}
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
{% endhighlight %}

### CUDA 설정하기
Theano가 GPU를 쓸 수 있도록 하기 위해서는 `CUDA_ROOT=/usr/local/cuda-7.5/`를 **.bashrc**같은 데에 추가해줘야 한다.
(Linux에서 CUDA를 설치하는 방법은 다른 [문서](http://localhost:4000/cuda/2016-02-29-cuda-linux/)를 참조하자)

사실, Debian이나 Ubuntu에서 CUDA를 설치하면 기본적으로 추가가 되어 있다.
이후 Theano에서 사용할 GPU 지정을 위해 다음과 같이 terminal에 입력한다.

{% highlight shell %}
echo '[global]
device = gpu
floatX = float32' > ~/.theanorc
{% endhighlight %}

### 설치 확인

[Theano 설치 문서](http://deeplearning.net/software/theano/tutorial/using_gpu.html#testing-theano-with-gpu)안내에 따라 다음의 내용을 가진 파일을 만든다.
{% highlight python %}
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
{% endhighlight %}

그리고 다음 명령어를 통해서 실행결과가 같은지, 특별한 오류가 발생했는지 확인해보면 된다.
{% highlight shell %}
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python check1.py
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python check1.py
{% endhighlight %}
