---
layout: post
title:  "Denoising Autoencoder (dA) with Theano"
date:   2016-04-25 0:20:00
categories: Deep Learning
description: Theano를 기반으로 구현한 Unsupervised Learning I
published: false
tags:
- DeepLearning
- Unsupervised learning
- Neural Networks
- Theano
- Auto-Encoder
---

참조:

1. [Theano Denoising Autoencoder (dA)](http://deeplearning.net/tutorial/dA.html)
2. [Wikipedia]()

Denoising Autoencoder(dA)는 **Autoencoder**의 확장된 형태로 다음과 같이 구성되어 있다.

<img class="col two center" src="/images/201604/Screen Shot 2016-04-25 at 2.52.01 PM.png"/>

따라서, **Autoencoder**에 대해서 알아보고, Corruption을 추가했을때 어떻게 Denosing Autoencoder가 구현되는지 살펴 볼 것이다.

# Autoencoder

Autoencoder는 unsupervised learning의 한종류로 입력 데이터의 representation을 학습키는 기법을 말한다. dimension reduction과 동일한 목적을 갖는다. 입력 unit의 수와 출력 unit의 수가 동일하며, 다른 어떤 데이터를 예측하도록 학습하는 것이 아니라 입력을 다시 복원할 수 있도록 학습하는 것을 특징으로 한다. Hidden Layer에서 입력에 대한 representation을 학습하므로 unsupervised learning이다.

Bengio 교수가 쓴 [Learning Deep Architecture for AI](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf)의 4.6절을 보면, *Autoencoder*에 대한 설명이 나와있는데, 필요한 내용은 Tutorial에 충분히 옮겨져 있다.

Autoencoder는 다음과 같은 구성을 갖고 있다.

<img class="col two center" src="/images/201604/Autoencoder_structure.png"/>

Theano에서는 1 hidden layered perceptron으로 구성되어 있으며 식으로 나타내면 다음과 같다.

**Encoding**

$$ y = s(\mathrm{Wx + b}) $$

$s$는 비선형의 activation 함수이며, 이 예제에서는 *sigmoid*를 이용했다.

**Reconstruction**

$$ z = s(\mathrm{W'y + b'}) $$

{% highlight python %}
class dA(object):
  """Denoising Auto-Encoder class (dA)
  """

  def __init__(
    self,
    numpy_rng,
    theano_rng=None,
    input=None,
    n_visible=784,
    n_hidden=500,
    W=None,
    bhid=None,
    bvis=None
  ):
    self.n_visible = n_visible
    self.n_hidden = n_hidden

    # Random Number Generation
    # create a Theano random generator that gives symbolic random values
    if not theano_rng:
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

    # note : W' was written as `W_prime` and b' as `b_prime`
    if not W:
        # W is initialized with `initial_W` which is uniformely sampled
        # from -4*sqrt(6./(n_visible+n_hidden)) and
        # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
        # converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        initial_W = numpy.asarray(
            numpy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                size=(n_visible, n_hidden)
            ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=initial_W, name='W', borrow=True)


    if not bvis:
        bvis = theano.shared(
            value=numpy.zeros(
                n_visible,
                dtype=theano.config.floatX
            ),
            borrow=True
        )

    if not bhid:
        bhid = theano.shared(
            value=numpy.zeros(
                n_hidden,
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

    self.W = W
    # b corresponds to the bias of the hidden
    self.b = bhid
    # b_prime corresponds to the bias of the visible
    self.b_prime = bvis
    # tied weights, therefore W_prime is W transpose
    self.W_prime = self.W.T
    self.theano_rng = theano_rng
    # if no input is given, generate a variable representing the input
    if input is None:
        # we use a matrix because we expect a minibatch of several
        # examples, each example being a row
        self.x = T.dmatrix(name='input')
    else:
        self.x = input

    self.params = [self.W, self.b, self.b_prime]
{% endhighlight %}

## Loss Function

autoencoder는 ANN의 출력이 입력을 복원하도록 하는 것이 특징이다. 따라서 loss 함수는 다음과 같다.

$$ L(\mathrm{xz}) = ||\mathrm{x - z}||^2 $$

여기서 입력 데이터가 bit vector이 vectors of bit probabilities(?)라면 cross entropy를 적용하여 다음과 같이 나타낼 수 있다.

$$ L(\mathrm{xz}) = - \sum _{k = 1} ^d [\mathrm{x}_k \mathrm{log} \mathrm{z}_k + (1 - \mathrm{x}_k) \mathrm{log}(1 - \mathrm{z}_k)]$$

{% highlight python %}
{ ... continue of dA class ... }
  def get_hidden_values(self, input):
    """ Computes the values of the hidden layer """
    return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

  def get_reconstructed_input(self, hidden):
    """Computes the reconstructed input given the values of the
    hidden layer

    """
    return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

  def get_cost_updates(self, corruption_level, learning_rate):
    """ This function computes the cost and the updates for one trainng
    step of the dA """

    y = self.get_hidden_values(self.x)
    z = self.get_reconstructed_input(y)
    # note : we sum over the size of a datapoint; if we are using
    #        minibatches, L will be a vector, with one entry per
    #        example in minibatch
    L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
    # note : L is now a vector, where each element is the
    #        cross-entropy cost of the reconstruction of the
    #        corresponding example of the minibatch. We need to
    #        compute the average of all these to get the cost of
    #        the minibatch
    cost = T.mean(L)

    # compute the gradients of the cost of the `dA` with respect
    # to its parameters
    gparams = T.grad(cost, self.params)
    # generate the list of updates
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(self.params, gparams)
    ]

    return (cost, updates)
{% endhighlight %}

## Training

매 입력마다 학습을 시키면서 입력값과 출력값을 비교하여 back-propagation을 통해 학습을 시키면 된다.

Theano 예제에서는 stochastic gradient descdent 방식으로 back-propagation을 수행한다.

한편 Wikipedia에서 다룬 [Autoencoder Training](https://en.wikipedia.org/wiki/Autoencoder#Training)에서는 pretraining([Restricted Boltzmann Machine?](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)]을 통해서 initial weight을 초기화 하여 learning을 할 경우에 learning의 성능을 높일 수 있다고 한다. 이것을 [Deep Brief Network](https://en.wikipedia.org/wiki/Deep_belief_network)라 한다.

## Tied Weights

Theano tutorial에서는 선택적이라고 말하면서, *tied weights*에 대해 소개하고 있다. 이는 reconstruction에서 사용될 weight matrix를 encoder의 weight matrix의 transpose 된 것이라고 간주하는 것을 말한다.

tied weight가 autoencoder에 가져다 주는 이점은 다음과 같다.

1. Learning parameter의 수 제한
1. Regularization

# Denoising Autoencoder

Hidden layer로 하여금 입력 데이터의 특징들을 학습하게 하는 것이 아니라 더 robust한 시스템을 만들기 위하여 입력데이터에 noise를 섞어, 이를 학습시킨다. 따라서 denosing autoencoder는 autoencoder의 구조에서 corruption을 하기 위한 절차를 추가하는 정도면 된다.

Theano 예제에서는 확률적으로 masking 하는 방법으로 corruption을 하는 방법을 사용한다.

{% highlight python %}
{ ... continue of dA Class ... }
  def get_corrupted_input(self, input, corruption_level):
    return self.theano_rng.binomial(size=input.shape, n=1,
                                    p=1 - corruption_level,
                                    dtype=theano.config.floatX) * input

  def get_cost_updates(self, corruption_level, learning_rate):
    """ This function computes the cost and the updates for one trainng
    step of the dA """

    # Input data에 대하여 Corruption을 하여 learning에 반영되도록 한다.
    tilde_x = self.get_corrupted_input(self.x, corruption_level)
    y = self.get_hidden_values(tilde_x)
    z = self.get_reconstructed_input(y)
    # note : we sum over the size of a datapoint; if we are using
    #        minibatches, L will be a vector, with one entry per
    #        example in minibatch
    L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
    # note : L is now a vector, where each element is the
    #        cross-entropy cost of the reconstruction of the
    #        corresponding example of the minibatch. We need to
    #        compute the average of all these to get the cost of
    #        the minibatch
    cost = T.mean(L)

    # compute the gradients of the cost of the `dA` with respect
    # to its parameters
    gparams = T.grad(cost, self.params)
    # generate the list of updates
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(self.params, gparams)
    ]

    return (cost, updates)
{% endhighlight %}
