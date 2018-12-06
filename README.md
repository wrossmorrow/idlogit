# idlogit (CURRENTLY IN DEVELOPMENT)

`idlogit` is a python "package" for estimating "idLogit" models, or Logit models with Idiosyncratic Deviations. The idLogit is a non-parametric model of choice heterogeneity with a convex maximum likelihood estimation problem. 

See [this article](https://web.stanford.edu/~morrowwr/idLogit) for methodological details. 

# Installing

As usual, do `pip install idlogit`. This package requires [`numpy`](http://www.numpy.org/), [`scipy`](https://www.scipy.org/), and [`ecos`](https://www.embotech.com/ECOS). 

# Using idlogit

## Basic Syntax

The most basic call is 

    x , info = idlogit( K , I , N , y , X , ind )

where

* `K` (integer) is the number of model features
* `I` (integer) is the number of individuals in the observations
* `N` (integer) is the number of observations
* `y` (numpy.array) is a N-vector of (binary) choices, coded as +/- 1
* `X` (numpy.array or scipy.sparse) is a NxK-matrix of observation-specific features (dense or sparse)
* `ind` (list or numpy.array) is a N-vector of observation-individual assignments in {1,...,I}

and 

* `x` (numpy.array) is a K-vector of estimated coefficients
* `info` (numpy.array) is the ECOS information structure resulting from the solve attempt

There are, of course, options we cover below. If a sparse `X` matrix is passed, it is internally transformed into a `scipy.sparse.coo_matrix` before use. If a dense `X` matrix is passed, it is _not_ processed as a sparse matrix; that is to say, `idlogit` presumes _all_ of `X`'s entries are nonzero. If this is not the case (for example, you have hard-coded dummies in the data) using a sparse matrix may be much better. 

## Options

Options that can currently be passed: 

* `constant` (boolean) Include a constant in the model, or not. The returned `x` will be `K+1` if `True`, with the first element being the estimated parameter corresponding to the constant. 
* `og` (boolean) Is there an "outside good", "outside option", or no-choice option? 
* `Lambdas` (list) A 2-element list of L1 and L2 penalty parameter values (respectively). 
* `bincat` (dict) A dictionary with fields `bin`, `cat` each of which is a list of indices from 1,...,K that identify which variables in `X` are _binary_ (0/1), which are _categorical_ (finite, with level-specific coefficients). Indices must be mutually exclusive. Binary variables are encoded with a single dummy equal to 1 for any "truthy" value in `X`. Categorical variables are analyzed for their cardinality and subsequently "expanded" into level-dummies whose coefficients are constrained to sum to zero for identification. 
* `prints` (dict) A dictionary of prints of the ECOS data created (for debugging, really). Valid keys are `start`, `costs`, `lineq`, `lerhs`, `cones`, `ccrhs`, and valid values are booleans (or anything "truthy").

as well as any options for [`ecos-python`](https://github.com/embotech/ecos-python) passed directly to ECOS as `**kwargs`. 

# Detailed Description

This code solves problems of the general form

    min 1/N sum_n log( 1 + exp{ -y_n x_n'( b + d_{i(n)} ) } ) + L1/N || d ||_1 + L2/2N || d ||_2
    wrt b , d_1 , ... , d_I in Real(K)
    sto d_1 + ... + d_I = 0

The solve is done by transforming this problem into an equivalent Exponential Cone Programming problem that can be passed to the [ECOS](https://www.embotech.com/ECOS) solver. 

# Contact

[W. Ross Morrow](mailto:morrowwr@gmail.com)