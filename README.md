# DimReduce

### Description

An implementation of 3 linear and non-linear well-known dimensionality reduction algorithms:  _Linear Discriminant Analysis_ (LDA), _Locality Sensitive Hashing_ (LSH) and _ISOMAP_.

### LDA

MATLAB implementation of Linear Discriminant Analysis (LDA), a well-known linear dimensionality reduction algorithm. The algorithm's utility is tested on a binary classification task applied on a benchmark spectrometry dataset of  potential cancer patients. The results, summarized in `html/Assignment.html`, demonstrate that LDA helps considerably in both
training time (since the dimensionality of the space is reduced from the original 10.000 to 2) and test-time accuracy when using a linear SVM and 10-fold cross validation.

### LSH

A Python implementation of Locality Sensitive Hashing (LSH) through random projections on the line of real numbers.  The algorithm is validated on an approximate nearest neighbor classification task by experimenting on a well-known Optical Character Recognition (OCR) dataset. It includes a a PDF README file of its own which details interesting
aspects of the implementation as well as practical insights regarding the algorithm.



### ISOMAP

A MATLAB implementation of ISOMAP, arguably the most well-studied Manifold Learning algorithm, is included in its namesake directory. The algorithm is validated on both synthetic 3D data as well as the data used and released by Tenenbaum, de Silva and Langford along with the [original paper](http://isomap.stanford.edu/). A more detailed README file in
PDF format is contained within the directory and provides additional details and figures.

License
-------

Refer to the file LICENSE for licensing details.


