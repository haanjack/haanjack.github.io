# Copyright (c) hanjack's LucidMind Project
# Distributed under the terms of the Modified BSD License.

FROM debian:latest
MAINTAINER hanjack <haanjack@gmail.com>

LABEL Description="This image is the base of python and octave app" Vendor="LucidMind Ltd." Version="1.0"

USER root

# Debian Update
RUN apt-get update && apt-get -y upgrade

# install Aptitude
RUN apt-get update && apt-get -y install --no-install-recommends\
  aptitude

# Install Utilities
RUN aptitude update && aptitude -y --without-recommends install\
  zsh git subversion vim curl wget unzip libhdf5-dev libpq-dev libzmq3-dev\
  libfreetype6-dev libpng-dev\
  liblapack-dev libblas-dev libatlas-base-dev gfortran libc6-dev\
  build-essential pkg-config &&\
  aptitude clean &&\
  aptitude autoclean &&\
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python 2
RUN aptitude update && aptitude -y --without-recommends install\
  python python-dev python-pip cython &&\
  aptitude -y install python-pandas python-h5py &&\
  aptitude clean &&\
  pip install\
    sympy

# Install Python 3
RUN aptitude update && aptitude -y --without-recommends install\
  python3 python3-dev python3-pip\
  aptitude -y install python-pandas python3-h5py &&\
  pip3 install\
    sympy statsmodels scikit_learn xlrd

# Install Utilities
RUN aptitude update && aptitude -y --without-recommends install\
  software-properties-common

# Install jupyter & kernel spec globally to avoid permission problems when NB_UID
# switching at runtime.
RUN pip install\
  jupyter &&\
  python2 -m pip install ipykernel &&\
  python2 -m ipykernel install --user &&\
  python3 -m pip install ipykernel &&\
  python3 -m ipykernel install --user &&\
  pip install git+https://github.com/dunovank/jupyter-themes.git &&\
  jupyter-theme -T -t space-legos -f Source-Code-Pro -fs 12

EXPOSE 8888

ADD notebook.sh /notebook.sh
RUN chmod a+x /notebook.sh

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

ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8

# Working Information
WORKDIR /workspace

CMD ["/bin/zsh"]
