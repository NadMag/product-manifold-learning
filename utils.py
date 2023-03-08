import numpy as np


def get_gt_data(data, datatype):
  '''
  converts the observed data to the ground truth data from the latent manifold
  '''
  if datatype == 'line_circle':
    data_gt = np.zeros((data.shape[0],2))
    data_gt[:,0] = data[:,0]
    data_gt[:,1] = np.arctan2(data[:,2], data[:,1])
  elif datatype == 'rectangle3d':
    data_gt = data[:,:2]
  elif datatype == 'torus':
    data_gt = np.zeros((data.shape[0],2))
    data_gt[:,0] = np.pi + np.arctan2(data[:,1] - np.mean(data[:,1]),
                                      data[:,0] - np.mean(data[:,0]))
    data_gt[:,1] = np.pi + np.arctan2(data[:,3] - np.mean(data[:,3]),
                                      data[:,2] - np.mean(data[:,2]))
  else:
    data_gt = data

  return data_gt


def get_product_eigs(manifolds, n_eigenvectors):
  '''
  returns the a list of product eigenvectors out of the first n_eigenvectors,
  given a manifold factorization
  manifolds: a list of lists, each sublist contains the indices of factor
             eigenvectors corresponding to a manifold factor
  n_eigenvectors: the number of eigenvectors
  '''
  mixtures = []
  for i in range(1, n_eigenvectors):
    is_mixture = True
    for manifold in manifolds:
      if i in manifold:
        is_mixture = False
    if is_mixture:
      mixtures.append(i)

  return mixtures


def print_manifolds(manifolds):
    for i,m in enumerate(manifolds):
      print("Manifold #{}".format(i + 1), m)
