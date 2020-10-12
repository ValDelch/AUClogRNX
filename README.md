# AUClogRNX

### Description

This code is an implementation of AUClogRNX, a quality measure for NLDR embeddings.

Given a dataset X in a high dimensional space and Y a low dimensional representation of X, one can compute

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Q_{NX}(K)=\frac{1}{KN}\sum_{i=1}^{N}|\nu_i^K\eta_i^K|" title="\Large Q_{NX}(K)=\frac{1}{KN}\sum_{i=1}^{N}|\nu_i^K\eta_i^K|" />
where 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\nu_i^K" title="\Large \nu_i^K" />
is the K-ary neighbourhoods of `x_i` in X, and 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\eta_i^K" title="\Large \eta_i^K" />
is the K-ary neighbourhoods of `y_i` in Y.

Then, AUClogRNX score is defined as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;R_{NX}(K)=\frac{(N-1)Q_{NX}(K)-K}{N-1-K}" title="\Large R_{NX}(K)=\frac{(N-1)Q_{NX}(K)-K}{N-1-K}" />
<img src="https://latex.codecogs.com/svg.latex?\Large&space;AUClogRNX=\frac{\sum_{K=1}^{N-2}\frac{R_{NX}(K)}{K}}{\sum_{K=1}^{N-2}\frac{1}{K}}" title="\Large AUClogRNX=\frac{\sum_{K=1}^{N-2}\frac{R_{NX}(K)}{K}}{\sum_{K=1}^{N-2}\frac{1}{K}}" />

### Dependencies

* Cython
* Numpy
* Scipy

### Installation

```python
>> python setup.py build_ext --inplace
```

### Utilisation

```python
import AUClogRNX_cython
X = np.random.rand(1000,50)
Y = np.random.rand(1000,2)

AUClogRNX = AUClogRNX_cython.compute(X, Y)
```
