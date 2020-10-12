# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

"""
---
Implementation of AUClogRNX, a quality measure for NLDR embeddings.
For more details on the measure, see Lee, J. A., Peluffo-Ordonez, D. H., & Verleysen, M. (2015). 
Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while
preserving both local and global structure. Neurocomputing, 169, 246-261.
---

---
Detailled structure

.
└── AUClogRNX_cython.pyx
    ├── cpdef compute
    ├── cdef logRNX
    └── cdef compute_QNX
---

---
This implementation has been initially written by Adrien Bibal (University of Namur) and 
modified by Valentin Delchevalerie (University of Namur).

last modification: 12 October 2020
---
"""


from scipy.spatial.distance import pdist, squareform
import numpy as np

cimport libc.math as cmath
cimport numpy as np
from libcpp.unordered_map cimport unordered_map
np.import_array()


cdef double[:] compute_QNX(np.ndarray[np.int_t, ndim=2] arr1, 
                           np.ndarray[np.int_t, ndim=2] arr2):
    """
    Compute QNX for each k in {1, ..., N-1}
    
    ----------
    Parameters
    ----------
    * arr1 : matrix, shape (N, N-1)
        
    * arr2 : matrix, shape (N, N-1)
        
    -------
    Returns
    -------
    * QNX : matrix, shape (N-1)
        QNX[k] for each k in {1, ..., N-1}
    """
    
    cdef:
        int count = 0
        int N = arr1.shape[0]
        int already_seen
        int i, j, k
        double[:] QNX = np.zeros((N-2))
        unordered_map[np.int_t, int] tmpset = unordered_map[np.int_t, int]()

    for i in range(N):
        idx = []
        already_seen = 0
        for k in range(1, N-1):
            # insert arr1[i, j] as key without assigned value
            tmpset[arr1[i, k-1]]
            idx.append(k-1)
            count = already_seen
            to_remove = []
            for j in idx:
                # check whether arr2[i, j] is in tmpset
                if tmpset.count(arr2[i, j]):
                    count += 1
                    already_seen += 1
                    to_remove.append(j)
            for j in to_remove:
                idx.remove(j)

            QNX[k-1] += count * cmath.pow(k*N, -1.0)
        tmpset.clear()
        
    return QNX


cdef double logRNX(np.ndarray[double, ndim=2] dataset, np.ndarray[double, ndim=2] projection):
    """
    Compute AUClogRNX score
    
    ----------
    Parameters
    ----------
    * arr1 : matrix, shape (N, dim1)
        data in the high dimensional space
        
    * arr2 : matrix, shape (N, dim2)
        data in the low dimensional space
        
    -------
    Returns
    -------
    * AUClogRNX : double
        AUClogRNX score
    """
    
    cdef:
        int N = projection.shape[0]
        int k
        double numerator = 0.0
        double denominator = 0.0
        double[:] QNX 
        np.ndarray[np.int_t, ndim=2] I_dataset, I_projection
        double[:,:] D_dataset, D_projection

    D_dataset    = squareform(pdist(dataset))
    D_projection = squareform(pdist(projection))

    I_dataset    = np.argsort(D_dataset, axis=1)[:, 1:]
    I_projection = np.argsort(D_projection, axis=1)[:, 1:]

    QNX = compute_QNX(I_projection[:, :], I_dataset[:, :])
    for k in range(1, N-1):
        numerator   += ((N-1) * QNX[k-1] - k) * cmath.pow((N-1-k)*k, -1.0)
        denominator += cmath.pow(k, -1.0)

    return numerator * cmath.pow(denominator, -1.0)


cpdef compute(np.ndarray[double, ndim=2] data, np.ndarray[double, ndim=2] visu):
    """
    Compute AUClogRNX
    
    ----------
    Parameters
    ----------
    * data : matrix, shape (N, dim1)
        data in the high dimensional space
        
    * visu : matrix, shape (N, dim2)
        data in the low dimensional space
        
    -------
    Returns
    -------
    * AUClogRNX : double
        Return the log of the AUC of K neighborhoods for a growing K
    """
    
    return logRNX(data, visu)
