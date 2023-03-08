from itertools import combinations
import numpy as np

from diffusion_maps import *
from factorization_graph import *
from utils import *

def factorize_manifold(
    data, 
    kernel_width, 
    n_eigenvectors, 
    n_factors, 
    eig_crit, 
    sim_crit,
    exclude_eigs=None
  ):
  """ Computes the approximate eigenfactors of a product manifold from a given sample of points.
  
  Args:
      data: A (N x D) matrix of  N sampled datapointsa of dimension D.
      kernel_width: (sigma) The width of the RBF kernel used to measure similarity of data points
      n_eigenvectors: Number of eigenvectors used to approximate the Laplacian
      n_factors: Number of factors of the manifolds
      eig_crit: Eigenvalue threshold used to filter bad factorizations
      sim_crit: Eigenvector similarity threshold used to filter bad factorizations
      exclude_eigs (optional): Egienvectors to exclude in factorization. Defaults to None.

  Returns:
      Dictionary: A results DTO containing intermediate computations and factorization.
  """  
  result = {}
  result['data'] = data

  # Approximating the Laplacian eigendecomposition
  similarty_mat = compute_similarity_matrix(data, kernel_width)
  egvecs, egvals = comp_laplacian_eigendec(similarty_mat, kernel_width, n_eigenvectors)

  result['egvecs'] = egvecs
  result['egvals'] = egvals

  # Finding candidate factorizations of the eigenvectors
  best_matches, best_sims, all_sims = find_eigproduct_candidate_factors(egvecs, egvals, n_factors, eig_crit, sim_crit, exclude_eigs=exclude_eigs)
 
  result['best_matches'] = best_matches
  result['best_sims'] = best_sims
  result['all_sims'] = all_sims

  # Partitioning the eigenfactors by their respective manifold
  labels, C = split_eigenvectors(best_matches, best_sims)
  result['C_matrix'] = C

  manifolds = []
  for m in range(n_factors):
    manifold_factor_indicies = labels[0][np.where(labels[1]==m)[0]]
    manifolds.append(manifold_factor_indicies)

  # make sure manifold with first nontrivial eigenvector comes first in list
  for idx,m in enumerate(manifolds):
    if 1 in m:
      m1 = manifolds.pop(idx)
      manifolds.insert(0, m1)
  result['manifolds'] = manifolds

  return result


def find_eigproduct_candidate_factors(egvecs, egvals, n_factors=2, eig_crit=10e-3, sim_crit=0.5, exclude_eigs=None):
  """ Returns all of the n_factors+1 tuples (product, ...factors) of possible factorizations of eigenvectors. 

  Args:
      egvecs: A (N x n_eigenvectors) matrix of the graph Laplacian eigenvectors. 
      egvals: A n_eigenvectors array of Laplacian eigenvalues.
      n_factors (int, optional): Number of factors in each product. Defaults to 2.
      eig_crit (_type_, optional): Difference criteria. Defaults to 10e-3.
      sim_crit (float, optional): Minimal similarity to retain. Defaults to 0.5.
      exclude_eigs (_type_, optional): A list of indices to not factor. Defaults to None.

  Returns:
      best_matches: Dictionary of triplets indexed by the product eigenvector
      max_sims: Dictionary of the triplet correlations indexed by the product eigenvector
      all_sims: All of the correlations for each product eigenvector 1...n_eigenvectors
  """  
  best_matches = {}
  max_sims = {}
  all_sims = {}

  # Iterate over product eigenvec
  # First eigenvector is the constant corresponding to 0
  for k in range(2, egvecs.shape[1]): 

    if (k % 10 == 0):
      print("[%-20s] %d%%" % ('='*int(20*k/egvecs.shape[1]), 100*k/egvecs.shape[1]))

    curr_eigenvec = egvecs[:,k]
    curr_eigenval = egvals[k]
    max_sim = 0
    best_match = []

    # Iterate over number of factors in product
    for m in range(2, n_factors + 1):
      if exclude_eigs is not None:
        valid_eigs_indices = [i for i in np.arange(1, k) if i not in exclude_eigs]
      else:
        valid_eigs_indices = np.arange(1, k)

      for combo in list(combinations(valid_eigs_indices, m)):
        combo = list(combo)
        egvals_sum = np.sum(egvals[combo])
        egvals_diff = abs(curr_eigenval - egvals_sum)
        if egvals_diff < eig_crit:
          product = np.prod(egvecs[:, combo], axis=1)
          sim = abs(cosine_similarity(product, curr_eigenvec))
          
          if sim > max_sim:
            best_match = combo
            max_sim = sim

    if len(best_match) > 0:
      all_sims[k] = max_sim
      if max_sim >= sim_crit:
        best_matches[k] = list(best_match)
        max_sims[k] = max_sim

  return best_matches, max_sims, all_sims
