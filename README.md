# idlogit

`idlogit` is a python "package" for estimating "idLogit" models, or Logit models with Idiosyncratic Deviations. The idLogit is a non-parametric model of choice heterogeneity with a convex maximum likelihood estimation problem. 

See [this article](https://web.stanford.edu/~morrowwr/idLogit) for methodological details. 

# Installing

As usual, do `pip install idlogit`. This package requires [`numpy`](http://www.numpy.org/), [`scipy`](https://www.scipy.org/), and [`ecos`](https://www.embotech.com/ECOS). 

# Using idlogit

The most basic call is 

    idlogit( K , I , N , y , X , ind )

where

* K (integer) is the number of model features
* I (integer) is the number of individuals in the observations
* N (integer) is the number of observations
* y (numpy.array) is a N-vector of +/- 1 coded choices
* X (numpy.array or scipy.sparse) is a NxK-matrix of observation-specific features (dense or sparse)
* ind (list or numpy.array) is a N-vector of observation-individual assignments in {1,...,I}

# Detailed Description

This code 

# Contact

[W. Ross Morrow](mailto:morrowwr@gmail.com)