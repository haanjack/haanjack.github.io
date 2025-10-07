---
layout: post
title:  "Jupyter 노트북 테마적용"
date:   2016-03-08 0:20:00
categories: Tool
published: false
tags:
- Jupyter
---

[Jupyter](http://jupyter.org)의 테마는 원래 하얀 색이다.
<img class="col" src="http://jupyter.org/assets/jupyterpreview.png"/>
<!-- ![Jupyter 홈페이지에서 볼 수 있는 Jupyter의 기본 화면](http://jupyter.org/assets/jupyterpreview.png) -->

코딩이 가능하면서도 문서와 같은 분위기여서 사용하는데 큰 무리 없이 사용할 수 있다. 하지만 모든 것(터미널 에디터 등)을 검은 화면으로 사용하는 나에게 하얀 화면은 너무 눈이 부셔서 어쩔 수 없이 보다 어두운 색으로 바꾸고 싶다는 생각이 들었다.

검색을 해보니 다음과 같은 github repository가 나왔다.

* [42 Jupyter theme](https://github.com/nsonnad/base16-ipython-notebook)
* [3 Jupyter-themes](https://github.com/dunovank/jupyter-themes)

두 가지를 모두 해보았는데, 최종적으로는 dunovank의 theme가 설치도 쉽고 색상 등에 있어서 더 가독성이 좋다는 생각이 들었다.

## 설치

{% highlight shell %}
pip install git+https://github.com/dunovank/jupyter-themes.git
{% endhighlight %}

그러면 **jupyter-theme**가 설치되고 옵션에 따라 jupyter의 생김새를 변경할 수 있게 된다.

## jupyter-theme의 옵션

| 종류 | 기능 | 예 |
|:---:|---|---|
| -l | 목록보기 | jupyter-theme -l |
| -t | 테마 선택 (oceans16 | grade3 | space-legos) | jupyter-theme -t oceans16 |
| -T | 툴바 보임 (기본: 툴바 없음) | jupyter-theme -T |
| -f | 폰트 선택 | jupyter-theme -f Source-Code-Pro |
| -fs | 폰트 크기 | jupyter-theme -fs 12 |
| -r | 초기화 | jupyter-theme -r |

여기서 툴바는, jupyter note의 최상위에 있는 jupyter 로고와 파일명이 쓰여진 부분을 말한다.
font는 자신의 사용하는 환경에 설치가 되어있는 font를 사용하여야 한다. 나의 경우에는 **[Source-Code-Pro](https://github.com/adobe-fonts/source-code-pro)**를 사랑하므로 이를 설치해두고 설정하였다.

## 테마 설치

{% highlight shell %}
jupyter-theme -t oceans16 -f Source-Code-Pro -fs 12
{% endhighlight %}

끝.. 이제 어두운 화면과, 어두운 에디터와 나란히 jupyter를 쓸 수 있게 되었다.
