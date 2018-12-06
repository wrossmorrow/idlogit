# idlogit

`idlogit` is a python "package" for estimating "idLogit" models, or Logit models with Idiosyncratic Deviations. The idLogit is a non-parametric model of choice heterogeneity with a convex maximum likelihood estimation problem. 

See [this article](https://web.stanford.edu/~morrowwr/idLogit) for methodological details. 

# Installing

As usual, do `pip install idlogit`. This package requires [`numpy`](http://www.numpy.org/), [`scipy`](https://www.scipy.org/), and [`ecos`](https://www.embotech.com/ECOS). 

# Using idlogit

The most basic call is 

    x , info = idlogit( K , I , N , y , X , ind )

where

* `K` (integer) is the number of model features
* `I` (integer) is the number of individuals in the observations
* `N` (integer) is the number of observations
* `y` (numpy.array) is a N-vector of +/- 1 coded choices
* `X` (numpy.array or scipy.sparse) is a NxK-matrix of observation-specific features (dense or sparse)
* `ind` (list or numpy.array) is a N-vector of observation-individual assignments in {1,...,I}

and 

* `x` (numpy.array) is a K-vector of estimated coefficients
* `info` (numpy.array) is the ECOS information structure resulting from the solve attempt

There are, of course, options we cover later. 

# Detailed Description

This code solves problems of the general form

    min 1/N sum_n log( 1 + exp{ -y_n x_n'( beta + delta_{i(n)} ) } ) + L1/N || delta ||_1 + L2/2N || delta ||_2
    wrt beta , delta_1 , ... , delta_I in Real(K)
    sto delta_1 + ... + delta_I = 0

The solve is done by transforming this problem into an equivalent Exponential Cone Programming problem that can be passed to the [ECOS](https://www.embotech.com/ECOS) solver. 

# Contact

[W. Ross Morrow](mailto:morrowwr@gmail.com)