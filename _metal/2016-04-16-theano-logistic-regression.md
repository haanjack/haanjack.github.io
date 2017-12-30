---
layout: post
title:  "Theano를 통한 Logical Regression"
date:   2016-04-15 0:40:00
categories: Deep Learning
description: Theano를 통한 Logistic Regression 구현
published: false
tags:
- DeepLearning
- Classification
- Theano
---

# Theano Logistic Regression 구현

Classification의 대표적인 예제인 MNIST를 통해서 Theano에서 구현하는 Logistic Regression을 살펴보자.
내용을 따라가다보면 Machine Learning의 어떻게 녹아들어가 있는지 살펴볼 수 있다.

## 라이브러리 로드
{% highlight python %}
from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
{% endhighlight %}

이번 예제에서는 위와 같은 library가 사용된다.
* pickle: 소스 데이터로 mnist의 pickled data를 사용한다.
* timeit: 함수의 동작시간을 측정하기 위해 사용된다.
* numpy: Theano에 데이터를 입출력하기 위해서 필요하다.
* teano, theano.tensor: Machine learning algorithm

## Theano를 사용한 Deep Learning과 Machine Learning의 차이

### Machine Learning 절차
1. Hypothesis Function
  $$ h_\theta(x) = g(\theta ^T x) $$

2. Cost Function

  $$ J(\theta) = \frac {1}{m} \sum _{i = 1} ^m Cost(h _\theta(x^{(i)}), y^{(i)})$$

3. Gradient Descent & Parameter Update
  각 Parameter 별로 gradient을 계산하여 Parameter를 업데이트 한다.

4. Prediction
  * Gradient에 의해 업데이트 된 Parameter로 Cost를 재계산 한다.
  * 모델 적용시, 수렴된 파라미터를 기반으로 새로운 데이터의 속성을 분류한다.

### Deep Learning 절차

1. Softmax regression
  Deep Learning에서는 softmax 함수를 이용하여 logistic regression을 수행한다. Softmax 함수는 여러 종류로 분류하는 것을 보다 일반화한 것이다. 출력이 pdf (probability density function)이므로, 총합은 1이며, 각 요소별 분류간 확률이 나온다.

  [Stanford Deep Learning Tutorial](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)을 보면 좀 더 자세한 수식이 나오지만, 따질려다보니 지금 다룰 만한 내용이 아닌 것 같아서 넘어갈 것이다.

  여하튼, Deep Learning은 다음과 같이 구성한다.

  $$ P(Y = i|x, W, b) = softmax_i(Wx+b) = \frac {e^{W_ix+b_i}}{\sum_je^{W_jx+b_j}}$$

  $$ y_{pred} = argmax_iP(Y=i|x,W,b) $$


2. Cost function
  Deep Learning도 Gradient Descent를 사용하며, Gradient 대상이 되는 Cost Funciton이 있으며, Tutorial을 보면 Machine Learning에서 사용하는 식과 동일하다.

3. Gradient Descent & Prameter Update
  Theano 같이 Symbolic programming을 하는 환경에서는 Gradient 연산을 symbolic으로 대체할 수 있다. 너무 좋다...

4. Prediction
  Machine Learning과 동일하다.

## LogisticRegression 클래스 구현

Logistic Regression을 수행하기 위한 Softmax Regression에 사용할 파라미터 및 함수로 구성된 Class를 생성한다.

### 파라미터 구성
{% highlight python %}
class LogisticRegression(object):
  def __init__(self, input, n_in, n_out):
    ####### start-snippet-1 ######
    # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
    self.W = theano.shared(
      value=numpy.zeros(
        (n_in, n_out),
        dtype=theano.config.floatX
      ),
      name='W',
      borrow=True
  )
  # initialize the biases b as a vector of n_out 0s
  self.b = theano.shared(
    value=numpy.zeros(
        (n_out,),
        dtype=theano.config.floatX
        ),
        name='b',
        borrow=True
      )

  # Classificion의 확률 분포 계산
  self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

  # softmax에 의해 계산된 확률 분포를 바탕으로 데이터 분류
  self.y_pred = T.argmax(self.p_y_given_x, axis=1)
  ####### end-snippet-1 #######

  # parameters of the model
  self.params = [self.W, self.b]

  # keep track of model input
  self.input = input
{% endhighlight %}

### Loss 함수
문서에서 보면, 'multi-class logistic regression에서 negative log-likelihood를 주로 사용한다'고 되어 있다. 이유는 안 써있다...;;

나는 **negative log-likelihood**도 모르니까 적당히 넘어가지 말고 찾아보기로 했다.
Googling을 해서 찾아본 [문서](https://quantivity.wordpress.com/2011/05/23/why-minimize-negative-log-likelihood/)를 보면,
likelihood는 조건부 확률을 의미하며 다음식과 같이 구성된다.

$$\mathcal{L}(\theta\,|\,x_1,\ldots,x_n) = f(x_1,x_2,\ldots,x_n|\theta) = \prod\limits_{i=1}^n f(x_i|\theta)$$

조건부 확률에서 여러 조건을 만족시키는 경우에 대한 식이니까 길기만 할뿐 별 건 아니다. 이것에 log를 하면 다음처럼 된다.

$$\log \mathcal{L}(\theta\,|\,x_1,\ldots,x_n) = \sum\limits_{i=1}^n \log f(x_i|\theta)$$

Log를 통해서 얻을 수 있는 이익은 다음과 같다.

1. Underflow를 방지할 수 있다
1. 곱셈을 덧셈으로 바꿔준다
1. log는 단조함수이므로, 쉽게 transform 할 수 있다.

이제 여기서 확률분포(pdf)를 생각해볼때, 가장 확률이 높은 조건을 [MLE(maximum likelihood estimator)](https://en.wikipedia.org/wiki/Maximum_likelihood)라고 한다.

$$ \hat{\theta}_{MLE} = \underset{\theta}{\arg\max} \sum\limits_{i=1}^n \log f(x_i|\theta) $$

이는 다음과 같은 관계를 충족시킨다.

$\underset{x}{\arg\max} (x)  = \underset{x}{\arg\min} (-x)$

따라서 deep learning tutorial에서 언급한 negative log-likelihood는 softmax의 결과인 확률 분포상에서 오차를 cost로 삼기 위해서 사용하는 연산인것 같다.

{% highlight python %}
... Continue of LogisticRegression ...

  def negative_log_likelihood(self, y):
    return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

{% endhighlight %}

negative_log_likelihood 함수는 theano 답게 symbolic expression으로 되어 있다.

음.. 수학적인 부분에 대해서는 좀 더 알아보고 내용을 가다듬어야 할 것 같다.

### Error

학습된 파라미터로 예측한 결과와 실제 값이 다른 비율을 반환한다. 이를 위해서 y에는 실제 값을 전달해준다.

{% highlight python %}
... Continue of LogisticRegression ...

def errors(self, y):
  # check if y has same dimension of y_pred
  if y.ndim != self.y_pred.ndim:
    raise TypeError(
      'y should have the same shape as self.y_pred',
      ('y', y.type, 'y_pred', self.y_pred.type)
    )
  # check if y is of the correct datatype
  if y.dtype.startswith('int'):
    # the T.neq operator returns a vector of 0s and 1s, where 1
    # represents a mistake in prediction
    return T.mean(T.neq(self.y_pred, y))
  else:
    raise NotImplementedError()

{% endhighlight %}

### LogisticRegression Class의 초기화

{% highlight python %}
x = T.matrix('x')
y = T.vector('y')

classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
cost = classifier.negative_log_likelihood(y)
{% endhighlight %}

MNIST의 이미지 크기가 $28 \times 28$이고, MNIST에서 사용되는 숫자의 종류는 10가지이므로 위 코드와 같이 입력한다.


## LoadData

[MNIST](http://yann.lecun.com/exdb/mnist/)를 위해서 제공되는 파일을 train/valid/test 각각 목적에 맞게 불러들인다.
MNIST의 데이터는 $28 \times 28$ 크기의 이미지이며, 손으로 쓰여진 그림이다. 각각의 이미지는 손으로 쓰여진 숫자(x)와 어떤 숫자인지(y) 알려준다.

좋은 점은 이미지의 크기나 이미지 구획에 대해서 고민할 필요없이 잘 정리돈 추출된 데이터 집합이므로, 데이터 엔지니어에 의해서 데이터가 정리된다든지 하는 작업을 건너뛰고 바로 Classification을 해볼 수 있다.

{% highlight python %}
def load_data(dataset):
  # Download the MNIST dataset if it is not present
  # MNIST 파일이 없는경우 다운을 받아준다. 코드에 있는 url 주소를 참조하여 직접 다운로드 받는다.
  data_dir, data_file = os.path.split(dataset)
  if data_dir == "" and not os.path.isfile(dataset):
      # Check if dataset is in the data directory.
      new_path = os.path.join(
          os.path.split(__file__)[0],
          "..",
          "data",
          dataset
      )
      if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
          dataset = new_path

  if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
      from six.moves import urllib
      origin = (
          'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
      )
      print('Downloading data from %s' % origin)
      urllib.request.urlretrieve(origin, dataset)

  print('... loading data')

  # Load the dataset
  # 압축된 파일을 풀고, pickle 되어 있는 파일을 읽어들인다.
  with gzip.open(dataset, 'rb') as f:
      try:
          train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
      except:
          train_set, valid_set, test_set = pickle.load(f)

  # GPU(CUDA) 메모리를 사용하기 위해 theano shared 메모리로 변환시켜주는 함수
  def shared_dataset(data_xy, borrow=True):
      """ Function that loads the dataset into shared variables

      The reason we store our dataset in shared variables is to allow
      Theano to copy it into the GPU memory (when code is run on GPU).
      Since copying data into the GPU is slow, copying a minibatch everytime
      is needed (the default behaviour if the data is not in a shared
      variable) would lead to a large decrease in performance.
      """
      data_x, data_y = data_xy
      shared_x = theano.shared(numpy.asarray(data_x,
                                             dtype=theano.config.floatX),
                               borrow=borrow)
      shared_y = theano.shared(numpy.asarray(data_y,
                                             dtype=theano.config.floatX),
                               borrow=borrow)
      # When storing data on the GPU it has to be stored as floats
      # therefore we will store the labels as ``floatX`` as well
      # (``shared_y`` does exactly that). But during our computations
      # we need them as ints (we use labels as index, and if they are
      # floats it doesn't make sense) therefore instead of returning
      # ``shared_y`` we will have to cast it to int. This little hack
      # lets ous get around this issue
      return shared_x, T.cast(shared_y, 'int32')

  test_set_x, test_set_y = shared_dataset(test_set)
  valid_set_x, valid_set_y = shared_dataset(valid_set)
  train_set_x, train_set_y = shared_dataset(train_set)

  rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
          (test_set_x, test_set_y)]
  return rval

{% endhighlight %}

## Learning Model

이제 gradient descendent를 이용하여, Learning Parameter를 Tunning 한다.
여기서 Theano의 가장 큰 특징은, 다른 언어를 사용해서 개발을 할 때는 직접 gradient descedent 식을 코드로 작성해서 사용해야하지만, Theano는 **grad** 함수에 cost function 만을 전달해주면 된다. 즉, gradient에 대해 구현할 필요가 없이 다음과 같이 code를 작성만 해주면 된다.

{% highlight python %}
g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)
{% endhighlight %}

계산된 gradient는 learning parameter에 적용되야 하며 어떻게 모델에 적용될 것인지 명세를 해주어야 한다.

{% highlight python %}
# specify how to update the parameters of the model as a list of
# (variable, update expression) pairs.
updates = [(classifier.W, classifier.W - learning_rate * g_W),
           (classifier.b, classifier.b - learning_rate * g_b)]
{% endhighlight %}

이제 실질적으로 learning을 수행하기 위한 **train_model** 함수를 작성한다.

{% highlight python %}
train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)
{% endhighlight %}

theano의 함수를 작성하는 방법이란게, 이런식으로 필요한 동작들을 parameter로 전달해 주는 것이다. theano function은 symbolic description으로 명세된 코드들이 인자로 전달되었을때 실제 code로 컴파일 되어 실제 동작할 수 있도록 해준다. python programming 처럼 parameter를 지정할 수 있으며, 이중에서 input과 output만을 제외하면 있어도 되고 없어도 된다.

train_model 함수를 구성하기 위해서는 다음과 같은 것들을 해야한다.

1. gradient 연산 및 learning parameter 업데이트
1. batch gradient를 수행하기 위한 동작

앞서 작성한 updates의 명세를 보면, cost의 gradient와 learning parameter인 W, b가 계산된다. 또한 cost의 gradient를 계산하기 위한 cost function을 계산하기 위해서는 입력데이터인 x, y가 주어져야 한다. 이때 이 예제에서는 batch gradient descent를 이용하므로, batch size 단위로 데이터를 입력할 수 있어야 하며, 이를 위해 train_model 함수는  batch index를 입력 파라미터로 사용한다.

## Test Model

learning parameter를 업데이트할 수 있는 것이 train_model 함수라면, 테스트 데이터를 바탕으로 예측된 값과 얼마나 차이가 나는지 확인 하는 것이 test 함수이다.
이때 얼마나 차이가 나는지 확인하기 위해서 앞서 생성한 LogisticRegression 클래스의 error 함수를 이용한다.

validation과 test는 데이터 셋이 다를 뿐 같은 동작을 하므로 다음과 같이 코드를 작성한다.

{% highlight python %}
test_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)
{% endhighlight %}

동작 방식은 앞서 **train_model** 함수와 동일하게 입력으로 주어진 batch index를 기준으로 주어진 데이터 셋에서 적절한 data를 입력받아 error값, 즉 학습된 파라미터를 바탕으로 얼마나 실제 값과 예측 값이 차이가 나는 비율을 확인한다.

## Processing

이제 만든 함수들을 하나로 묶어보자.

### Loading Data
{% highlight python %}
def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
  datasets = load_data(dataset)

  train_set_x, train_set_y = datasets[0]
  valid_set_x, valid_set_y = datasets[1]
  test_set_x, test_set_y = datasets[2]

  # compute number of minibatches for training, validation and testing
  n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
  n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
  n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
{% endhighlight %}

### build model

지금까지 설명한 부분들을 코드로 작성한다. 입력에 사용될 parameter들에 대하여(batch index, x, y)에 대하여 사용할 수 있도록 theano의 변수로 정의해준다.

{% highlight python %}
{ ... continue of sgd_optimization_mnist ... }

  print('... building the model')

  # allocate symbolic variables for the data
  index = T.lscalar()  # index to a [mini]batch

  # generate symbolic variables for input (x and y represent a
  # minibatch)
  x = T.matrix('x')  # data, presented as rasterized images
  y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

  # construct the logistic regression class
  # Each MNIST image has size 28*28
  classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

  # the cost we minimize during training is the negative log likelihood of
  # the model in symbolic format
  cost = classifier.negative_log_likelihood(y)

  # compiling a Theano function that computes the mistakes that are made by
  # the model on a minibatch
  test_model = theano.function(
      inputs=[index],
      outputs=classifier.errors(y),
      givens={
          x: test_set_x[index * batch_size: (index + 1) * batch_size],
          y: test_set_y[index * batch_size: (index + 1) * batch_size]
      }
  )

  validate_model = theano.function(
      inputs=[index],
      outputs=classifier.errors(y),
      givens={
          x: valid_set_x[index * batch_size: (index + 1) * batch_size],
          y: valid_set_y[index * batch_size: (index + 1) * batch_size]
      }
  )

  # compute the gradient of cost with respect to theta = (W,b)
  g_W = T.grad(cost=cost, wrt=classifier.W)
  g_b = T.grad(cost=cost, wrt=classifier.b)

  # start-snippet-3
  # specify how to update the parameters of the model as a list of
  # (variable, update expression) pairs.
  updates = [(classifier.W, classifier.W - learning_rate * g_W),
             (classifier.b, classifier.b - learning_rate * g_b)]

  # compiling a Theano function `train_model` that returns the cost, but in
  # the same time updates the parameter of the model based on the rules
  # defined in `updates`
  train_model = theano.function(
      inputs=[index],
      outputs=cost,
      updates=updates,
      givens={
          x: train_set_x[index * batch_size: (index + 1) * batch_size],
          y: train_set_y[index * batch_size: (index + 1) * batch_size]
      }
  )
  # end-snippet-3
{% endhighlight %}

### TRAIN MODEL

이제 loop을 돌면서 보다 learning parameter를 update한다. 그리고 validation_frequency로 계산된 주기에 맞춰서 validation 코드를 실행한다.

이 예제에서는 원하는 조건이 달성 되었을 경우 학습을 종료할 수 있는 early termination에 대한 코드도 추가되어 있다.

{% highlight python %}
{ ... continue of sgd_optimization_mnist ... }

  print('... training the model')
  # early-stopping parameters
  patience = 5000  # look as this many examples regardless
  patience_increase = 2  # wait this much longer when a new best is
                                # found
  improvement_threshold = 0.995  # a relative improvement of this much is
                                # considered significant
  validation_frequency = min(n_train_batches, patience // 2)
                                # go through this many
                                # minibatche before checking the network
                                # on the validation set; in this case we
                                # check every epoch

  best_validation_loss = numpy.inf
  test_score = 0.
  start_time = timeit.default_timer()

  done_looping = False
  epoch = 0
  while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
      minibatch_avg_cost = train_model(minibatch_index)
      # iteration number
      iter = (epoch - 1) * n_train_batches + minibatch_index

      if (iter + 1) % validation_frequency == 0:
        # compute zero-one loss on validation set
        validation_losses = [validate_model(i)
                             for i in range(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)

        print(
            'epoch %i, minibatch %i/%i, validation error %f %%' %
            (
                epoch,
                minibatch_index + 1,
                n_train_batches,
                this_validation_loss * 100.
            )
        )

        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
          #improve patience if loss improvement is good enough
          if this_validation_loss < best_validation_loss *  \
             improvement_threshold:
              patience = max(patience, iter * patience_increase)

          best_validation_loss = this_validation_loss
          # test it on the test set

          test_losses = [test_model(i)
                         for i in range(n_test_batches)]
          test_score = numpy.mean(test_losses)

          print(
              (
                  '     epoch %i, minibatch %i/%i, test error of'
                  ' best model %f %%'
              ) %
              (
                  epoch,
                  minibatch_index + 1,
                  n_train_batches,
                  test_score * 100.
              )
          )

          # save the best model
          with open('best_model.pkl', 'wb') as f:
              pickle.dump(classifier, f)

      if patience <= iter:
        done_looping = True
        break

  end_time = timeit.default_timer()
  print(
      (
          'Optimization complete with best validation score of %f %%,'
          'with test performance %f %%'
      )
      % (best_validation_loss * 100., test_score * 100.)
  )
  print('The code run for %d epochs, with %f epochs/sec' % (
      epoch, 1. * epoch / (end_time - start_time)))
  print(('The code for file ' +
         os.path.split(__file__)[1] +
         ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)
{% endhighlight %}
