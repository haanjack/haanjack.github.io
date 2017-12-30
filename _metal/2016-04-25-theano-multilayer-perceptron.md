---
layout: post
title:  "Theano를 통한 Multilayer Perceptrion"
date:   2016-04-25 0:10:00
categories: Deep Learning
description: Theano를 통한 Neural Networks
published: false
tags:
- DeepLearning
- Neural Networks
- Theano
---

앞서 MNIST 예제에서 사용한 Neuarl Network Model에서 사용한 모델은 matrix 연산을 기반으로 machine learning을 한 것으로 아직 deep learning이라고 하기에는 부족다. 이번 에제는 하나의 hidden layer를 갖는 MLP(Multi-Layer Perceptron)을 이용하여 Machine Learning을 해보도록 하자. 여기서부터가 비로소 ANN(Aritificial Neural Network) 또는 Deep Learning이라고 부른다.

# Model

MLP를 표현하는 방법은 크게 두 가지 방법이 있는데, 하나는 graph를 사용한 것이고 하나는 수학적인 표현으로 나타내는 것이다.

<div class="img_row center">
<img src="/images/201604/mlp.png"/>
<img class="margin-top: auto; margin-bottom: auto;" src="/images/201604/3532b34fba1f5111baf8f41e241768fcafa23540.png"/>
</div>
<div class="col three caption">
hidden layer가 하나인 경우에 대하여 MLP를 표현한 예. 하나의 layer가 진행할 때마다 하나의 polynomial equation을 한다고 볼 수 있다.
</div>

수식으로 표현 된 Neural Network을 보면, bias vector인 $b^{(1)}$, $b^{(2)}$, Weight matrix인 $W^{(1)}$, $W^{(2)}$, 그리고 activation function인 $G$와 $s$로 구성되어 있음을 확인할 수 있다. 여기서 $ h(x) = s(b^{(1)} + W^{(1)} x) $는 hidden layer를 표현하며, output layer는 $ o(x) = G(b^{(2)} + W^{(2)} x) $로 표현된다. Activation Function $s$는 보통 $tanh$, $sigmoid$, $ReLU$ 등의 함수들이 사용된다.

<div class="img_row">
  <img class="col two display: inline-block;" src="/images/201604/700px-Gjl-t(x).svg.png"/>
  <img class="col one" src="/images/201604/activation_funcs1.png"/>
</div>
<div class="col three caption">
  여러 Activation 함수들의 특징. logical classification에 적절한 동작을 할 수 있도록 해준다.
</div>

Theano 예제에서는 $tanh$를 사용하겠다고 하였으며, 이는 학습속도가 빠르기 때문이라고 한다. coursera 강의에서는 $sigmoid$ 함수를 이용하며, udacity의 tensorflow강의에서는 $ReLU$ 함수를 이용한다. 개발하면서 적절한 것을 선택해서 사용하면 되는 문제로 보인다.

output vector는 $o(x) = G(b^{(2)} + W^{(2)} h(x))$의 연산을 통해 구할 수 있다. 이 연산은 앞서 다룬 MNIST Logistic Regression에서 사용하는 연산과 동일한 연산이며, activation 함수 $G$로는 $softmax$를 이용한다.

# Hidden Layer Class 구성

MLP를 구성하는데 사용할 Hidden Layer를 구성하는데 코드를 재사용하기 위하여 Hidden Layer class를 구성한다.

{% highlight python %}
class HiddenLayer(object):
  def __init__(self, rng, input, n_in, n_out, W=None, b=None,
               activation=T.tanh):
    self.input = input

    # Weight matrix 초기화
    if W is None:
      W_values = numpy.asarray(
        rng.uniform(
          low=-numpy.sqrt(6. / (n_in + n_out)),
          high=numpy.sqrt(6. / (n_in + n_out)),
          size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
      )
      if activation == theano.tensor.nnet.sigmoid:
          W_values *= 4

      W = theano.shared(value=W_values, name='W', borrow=True)

    # bias vector 초기화
    if b is None:
      b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
      b = theano.shared(value=b_values, name='b', borrow=True)

    self.W = W
    self.b = b

    lin_output = T.dot(input, self.W) + self.b
    self.output = (
      lin_output if activation is None
      else activation(lin_output)
    )
    # parameters of the model
    self.params = [self.W, self.b]
{% endhighlight %}

여기서 Weight Matrix에 random number를 임의로 주지 않고 uniformly distribution한 random number로서, 범위를 정해서 주는 이유는 activation function의 유효한 범위 때문이라고 한다.

각각의 activation 함수에 따른 유효 범위는 다음과 같다.

| | $tanh$ | $sigmoid$ | $LeLU$ |
|:---:|:---:|:---:|:---:|
|low| $- \sqrt {\frac {6}{n_{in} + n_{out}}}$ | $-4\sqrt {\frac {6}{n_{in} + n_{out}}}$ | (?) |
|high| $\sqrt {\frac {6}{n_{in} + n_{out}}}$ | $4\sqrt {\frac {6}{n_{in} + n_{out}}}$ | (?) |


# MLP Class 구성

이제 Hidden layer와 ouput layer를 묶어서 하나의 MLP로 구성할 것이다.
output layer는 MNIST logistic regression과 동일한 구성을 갖는다. 이 예제는 하나의 hidden layer만을 사용하므로, MLP의 paramter 중 hidden layer의 unit의 수를 지정하는 *n_hidden*에 대해서도 하나만 존재한다. 이 부분에 대해서는, hidden layer가 여러개가 되었을 때엔 array로 지정할 수 있도록 변형 될 것이다.

{% highlight python %}
class MLP(object):
  """Multi-Layer Perceptron Class"""

  def __init__(self, rng, input, n_in, n_hidden, n_out):
    """Initialize the parameters for the multilayer perceptron"""

    # Hidden Layer 정의
    self.hiddenLayer = HiddenLayer(
        rng=rng,
        input=input,
        n_in=n_in,
        n_out=n_hidden,
        activation=T.tanh
    )

    # Output Layer에 정의. MNIST 예제에서 사용한 Logistic Regression을 적용한다.
    self.logRegressionLayer = LogisticRegression(
        input=self.hiddenLayer.output,
        n_in=n_hidden,
        n_out=n_out
    )

    # Regularization Opetions
    # L1 norm
    # one regularization option is to enforce L1 norm to be small
    self.L1 = (
        abs(self.hiddenLayer.W).sum()
        + abs(self.logRegressionLayer.W).sum()
    )

    # square of L2 norm ;
    # one regularization option is to enforce square of L2 norm to be small
    self.L2_sqr = (
        (self.hiddenLayer.W ** 2).sum()
        + (self.logRegressionLayer.W ** 2).sum()
    )

    # Output layer의 cost 함수를 MLP의 cost 함수로 지정한다.
    self.negative_log_likelihood = (
        self.logRegressionLayer.negative_log_likelihood
    )
    # 예측한 값의 오류를 확인하고자 하는 errors의 함수도 MLP와 output layer에 동일하게 적용한다.
    self.errors = self.logRegressionLayer.errors

    # the parameters of the model are the parameters of the two layer it is
    # made out of
    self.params = self.hiddenLayer.params + self.logRegressionLayer.params
    # end-snippet-3

    # keep track of model input
    self.input = input
{% endhighlight %}

# Processing

이제 Symbolic Expresion으로 명세한 MLP를 기반으로 Deep Learning을 해보자.

{% highlight python %}
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
  """
  Demonstrate stochastic gradient descent optimization for a multilayer
  perceptron

  This is demonstrated on MNIST.

  :type learning_rate: float
  :param learning_rate: learning rate used (factor for the stochastic
  gradient

  :type L1_reg: float
  :param L1_reg: L1-norm's weight when added to the cost (see
  regularization)

  :type L2_reg: float
  :param L2_reg: L2-norm's weight when added to the cost (see
  regularization)

  :type n_epochs: int
  :param n_epochs: maximal number of epochs to run the optimizer

  :type dataset: string
  :param dataset: the path of the MNIST dataset file from
               http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


 """
  # Data Load
  datasets = load_data(dataset)

  train_set_x, train_set_y = datasets[0]
  valid_set_x, valid_set_y = datasets[1]
  test_set_x, test_set_y = datasets[2]

  # 사용할 데이터의 batch의 개수 계산
  n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
  n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
  n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

  ######################
  # BUILD ACTUAL MODEL #
  ######################
  print('... building the model')

  # allocate symbolic variables for the data
  index = T.lscalar()  # index to a [mini]batch
  x = T.matrix('x')  # the data is presented as rasterized images
  y = T.ivector('y')  # the labels are presented as 1D vector of
                      # [int] labels

  rng = numpy.random.RandomState(1234)

  # construct the MLP class
  classifier = MLP(
      rng=rng,
      input=x,
      n_in=28 * 28,
      n_hidden=n_hidden,
      n_out=10
  )

  # cost function;
  # linear regression에 대한 것도 고려한 cost 함수를 사용한다.
  cost = (
      classifier.negative_log_likelihood(y)
      + L1_reg * classifier.L1
      + L2_reg * classifier.L2_sqr
  )
  # end-snippet-4

  # test, validation 함수 정의. MINST 예제와 동일하다.
  test_model = theano.function(
      inputs=[index],
      outputs=classifier.errors(y),
      givens={
          x: test_set_x[index * batch_size:(index + 1) * batch_size],
          y: test_set_y[index * batch_size:(index + 1) * batch_size]
      }
  )

  validate_model = theano.function(
      inputs=[index],
      outputs=classifier.errors(y),
      givens={
          x: valid_set_x[index * batch_size:(index + 1) * batch_size],
          y: valid_set_y[index * batch_size:(index + 1) * batch_size]
      }
  )

  # MLP의 각 layer마다 gradient를 계산한다.
  gparams = [T.grad(cost, param) for param in classifier.params]

  # gradient연산은 update와 함께 연동되어 learning parameter를 gradient 계산과 함께 더한다.
  updates = [
      (param, param - learning_rate * gparam)
      for param, gparam in zip(classifier.params, gparams)
  ]

  # train model 함수 역시 MINST 예제와 동일하게 명세한 update 함수를 적용하여 사용한다.
  train_model = theano.function(
      inputs=[index],
      outputs=cost,
      updates=updates,
      givens={
          x: train_set_x[index * batch_size: (index + 1) * batch_size],
          y: train_set_y[index * batch_size: (index + 1) * batch_size]
      }
  )

  ###############
  # TRAIN MODEL #
  ###############
  print('... training')

  # early-stopping parameters
  patience = 10000  # look as this many examples regardless
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
  best_iter = 0
  test_score = 0.
  start_time = timeit.default_timer()

  epoch = 0
  done_looping = False

  while (epoch < n_epochs) and (not done_looping):
      epoch = epoch + 1
      for minibatch_index in range(n_train_batches):

          minibatch_avg_cost = train_model(minibatch_index)
          # iteration number
          iter = (epoch - 1) * n_train_batches + minibatch_index

          if (iter + 1) % validation_frequency == 0:
              # compute zero-one loss on validation set
              validation_losses = [validate_model(i) for i
                                   in range(n_valid_batches)]
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
                  if (
                      this_validation_loss < best_validation_loss *
                      improvement_threshold
                  ):
                      patience = max(patience, iter * patience_increase)

                  best_validation_loss = this_validation_loss
                  best_iter = iter

                  # test it on the test set
                  test_losses = [test_model(i) for i
                                 in range(n_test_batches)]
                  test_score = numpy.mean(test_losses)

                  print(('     epoch %i, minibatch %i/%i, test error of '
                         'best model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

          if patience <= iter:
              done_looping = True
              break

  end_time = timeit.default_timer()
  print(('Optimization complete. Best validation score of %f %% '
         'obtained at iteration %i, with test performance %f %%') %
        (best_validation_loss * 100., best_iter + 1, test_score * 100.))
  print(('The code for file ' +
         os.path.split(__file__)[1] +
         ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
{% endhighlight %}
