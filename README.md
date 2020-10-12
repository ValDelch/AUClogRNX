# AUClogRNX

### Description

This code is an implementation of AUClogRNX, a quality measure for NLDR embeddings.

Given a dataset $`X`$ in a high dimensional space and $`Y`$ a low dimensional representation of $`X`$, one can compute
```math
Q_{NX}(K) = \frac{1}{KN} \sum_{i=1}^{N} |\nu_i^K \eta_i^K|
```
<img src="https://latex.codecogs.com/svg.latex?\Large&space;Q_{NX}(K) = \frac{1}{KN} \sum_{i=1}^{N} |\nu_i^K \eta_i^K|" title="\Large Q_{NX}(K) = \frac{1}{KN} \sum_{i=1}^{N} |\nu_i^K \eta_i^K|" />

where $`\nu_i^K`$ is the K-ary neighbourhoods of $`x_i`$ in $`X`$, and $`\eta_i^K`$ is the K-ary neighbourhoods of $`y_i`$ in $`Y`$

### Dependencies

* Cython
* Numpy
* Scipy

### Installation

```python
>> python setup.py build_ext --inplace
```

### Verification

If matplotlib is installed, you can try...

```python
>> python test_PM.py
```

... in order to test the implementation on the 70 000 instances of MNIST.

### Utilisation

```python
import PM_tSNE
tsne = PM_tSNE.PM_tSNE(n_iter=750, coeff=8.0, grid_meth='NGP')
# Load the 70 000 instances of MNIST (already prepared for t-SNE: standardized + reduced to 50 features with PCA)
X = np.load('./MNIST_data.npy', allow_pickle=True)
Embedding = tsne.fit_transform(X)
```
