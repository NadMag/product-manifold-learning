import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.extmath import randomized_svd

def compute_similarity_matrix(data, sigma):
  """Builds the Gaussian similarity matrix of given data points

  Args:
      data: A (N x D) matrix of  N sampled datapointsa of dimension D.
      sigma: The kernel width

  Returns:
      Returns a ``N * (N-1) / 2`` (N choose 2) sized vector `v` where :math:`v[{n \\choose 2} - {n-i \\choose 2} + (j-i-1)]` is the simillarity between points ``i`` and ``j``.
  """  
  pairwise_sq_dists = squareform(pdist(data, 'sqeuclidean'))
  W = np.exp(-pairwise_sq_dists / sigma)
  return W

def comp_laplacian_eigendec(smilarity_mat, sigma, n_eigenvectors):
  """Computes the eigendecomposition of the graph laplacian

  Args:
      similarity_mat: A ``N * (N-1) / 2`` (N choose 2) sized vector `v` where :math:`v[{n \\choose 2} - {n-i \\choose 2} + (j-i-1)]` is the simillarity between points ``i`` and ``j``.
      sigma: The similarity kernel width
      n_eigenvectors: The number of eigenectors to compute (trunction limit).

  Returns:
      eigenvecs: A (N x n_eigenvectors) matrix of the graph Laplacian eigenvectors.
      laplacian_egvals: A n_eigenvectors array of the laplacian approximated eigenvalues.
  """  
  ones = np.ones(smilarity_mat.shape[0])
  D_sqrt_diag = np.sqrt(smilarity_mat @ ones)
  L_sym = smilarity_mat / np.outer(D_sqrt_diag, D_sqrt_diag) #computational trick
  V, S, VT = randomized_svd(L_sym,
                                n_components=n_eigenvectors,
                                n_iter=5,
                                random_state=None)
  eigenvecs = V / np.reshape(V[:,0], (-1, 1)) # Standard eigenmap normalization
  laplacian_egvals = -np.log(S) / sigma
  return eigenvecs, laplacian_egvals