# Code taken from the original implementation: https://github.com/sxzhang25/product-manifold-learning/
# helper methods for creating figures

import numpy as np
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from scipy.linalg import block_diag
from factorization_graph import cosine_similarity

matplotlib.use('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from utils import *


def plot_synthetic_data(data, dimensions, title=None, filename=None,
                        azim=60, elev=30, proj_type='persp'):
  '''
  displays the observed data
  '''
  fig = plt.figure(figsize=(5,5))
  if data.shape[1] <= 2:
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0], data[:,1], s=5)
  elif data.shape[1] == 3:
    ax = fig.add_subplot(111, projection='3d', azim=azim, elev=elev,
                         proj_type=proj_type)
    x, y, z = dimensions
    axis_max = np.max(dimensions)
    ax.set_xlim3d((x - axis_max) / 2, (x + axis_max) / 2)
    ax.set_ylim3d((y - axis_max) / 2, (y + axis_max) / 2)
    ax.set_zlim3d((z - axis_max) / 2, (z + axis_max) / 2)
    ax.scatter(data[:,0], data[:,1], data[:,2], s=5, c=data[:,2])
  else:
    ax = fig.add_subplot(111, projection='3d', azim=azim, elev=elev,
                         proj_type=proj_type)
    x, y, z = dimensions
    axis_max = np.max(dimensions)
    ax.set_xlim3d((x - axis_max) / 2, (x + axis_max) / 2)
    ax.set_ylim3d((y - axis_max) / 2, (y + axis_max) / 2)
    ax.set_zlim3d((z - axis_max) / 2, (z + axis_max) / 2)
    g = ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3], s=5)
    cb = plt.colorbar(g)
  if title:
    plt.title(title, pad=10)
  if filename:
    plt.savefig(filename)
  plt.close()

def plot_cryo_em_data(data, title=None, filename=None):
  '''
  displays the cryo-EM images
  (be careful to only select a few to show)
  '''
  n_samples = data.shape[0]
  rows = int(np.ceil(n_samples**0.5))
  cols = rows
  fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
  for r in range(rows):
    for c in range(cols):
      ax = axs[r, c]
      index = r * cols + c
      if index >= n_samples:
        ax.set_visible(False)
      else:
        vol = data[index]
        ax.imshow(vol, cmap='gray')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
  if title:
    plt.title(title, pad=10)
  if filename:
    plt.savefig(filename)
  plt.close()

###
# PLOT EIGENVECTORS
###

def plot_eigenvector(data, v, scaled=False, title=None, filename=None):
  '''
  plots the eigenvector v
  if scaled=True, then scale v to [0,1]
  if index is specified, label the graph
  '''

  fig = plt.figure()
  ax = fig.add_subplot(111)
  g = ax.scatter(data[:,0], data[:,1], marker="s", c=v)
  ax.set_aspect('equal', 'datalim')
  cb = plt.colorbar(g)
  if title:
    plt.title(title)
  if filename:
    plt.savefig(filename)
  plt.close()

def plot_eigenvectors(data, dimensions, eigenvectors, full=True, labels=None, title=None,
                      filename=None, offset_scale=0.1, azim=60, elev=30, proj_type='persp'):
  '''
  plots the array eigenvectors

  data: the data corresponding to the components of the eigenvector
  dimensions: the dimensions of the data (for configuring axes)
  eigenvectors: a list of eigenvectors
  full: if set to True, each eigenvector will be displayed separately. Otherwise,
  they will be stacked.
  offset_scale: if full=True, then this determines the vertical spacing between
  stacked eigenvectors.
  '''
  rows = int(np.ceil(len(eigenvectors)**0.5))
  cols = rows
  if data.shape[1] <= 2:
    if full:
      fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
      for r in range(rows):
        for c in range(cols):
          ax = axs[r, c] if len(eigenvectors) > 1 else axs
          index = r * cols + c
          if index >= len(eigenvectors):
            ax.set_visible(False)
          else:
            v = eigenvectors[r * cols + c]
            v /= np.linalg.norm(v)
            g = ax.scatter(data[:,0], data[:,1], marker="s", c=v)
            fig.colorbar(g, ax=ax)
            ax.set_aspect('equal', 'datalim')
            if labels is not None:
              ax.set_title(labels[index])
    else:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d', azim=azim, elev=elev,
                           proj_type=proj_type)
      x, y, z = dimensions
      axis_max = np.max(dimensions)
      offset = offset_scale * len(eigenvectors)
      ax.set_xlim3d((x - axis_max) / 2, (x + axis_max) / 2 + offset)
      ax.set_ylim3d((y - axis_max) / 2, (y + axis_max) / 2 + offset)
      ax.set_zlim3d(0, len(eigenvectors))
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_zticks([])
      plt.axis('off')
      plt.grid(b=None)
      plt.tight_layout()
      plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

      for i in range(len(eigenvectors)):
        v = eigenvectors[i]
        v /= np.linalg.norm(v)
        g = ax.scatter(data[:,0] + offset_scale * np.ones(data.shape[0]) * (len(eigenvectors) - i),
                       data[:,1] + offset_scale * np.ones(data.shape[0]) * (len(eigenvectors) - i),
                       np.ones(data.shape[0]) * (len(eigenvectors) - i),
                       marker="s",
                       c=v)
  else:
    if full:
      fig = plt.figure(figsize=(5 * cols, 5 * rows))
      x, y, z = dimensions
      axis_max = np.max(dimensions)
      offset = offset_scale * len(eigenvectors)
      plt.axis('off')
      plt.grid(b=None)
      plt.tight_layout()
      for r in range(rows):
        for c in range(cols):
          index = r * cols + c
          ax = fig.add_subplot(rows, cols, index+1, projection='3d')
          ax.set_xlim3d((x - axis_max) / 2, (x + axis_max) / 2)
          ax.set_ylim3d((y - axis_max) / 2, (y + axis_max) / 2)
          ax.set_zlim3d((z - axis_max) / 2, (z + axis_max) / 2)
          ax.set_xticks([])
          ax.set_yticks([])
          ax.set_zticks([])
          if index >= len(eigenvectors):
            ax.set_visible(False)
          else:
            v = eigenvectors[r * cols + c]
            v /= np.linalg.norm(v)
            g = ax.scatter(data[:,0], data[:,1], data[:,2], marker="s", c=v)
            if labels is not None:
              ax.set_title(labels[index])
    else:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d', azim=azim, elev=elev,
                           proj_type=proj_type)
      x, y, z = dimensions
      axis_max = np.max(dimensions)
      offset = offset_scale * len(eigenvectors) + x
      ax.set_xlim3d(0, x * (len(eigenvectors) + offset))
      ax.set_ylim3d((y - axis_max) / 2, (y + axis_max) / 2 + offset)
      ax.set_zlim3d((z - axis_max) / 2, (z + axis_max) / 2 + offset)
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_zticks([])
      plt.axis('off')
      plt.grid(b=None)
      plt.tight_layout()
      plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

      for i in range(len(eigenvectors)):
        v = eigenvectors[i]
        v /= np.linalg.norm(v)
        g = ax.scatter(data[:,0] + offset_scale * np.ones(data.shape[0]) * (len(eigenvectors) - i)  + (np.ones(data.shape[0]) + x) * (len(eigenvectors) - i),
                       data[:,1] + offset_scale * np.ones(data.shape[0]) * (len(eigenvectors) - i),
                       data[:,2] + offset_scale * np.ones(data.shape[0]) * (len(eigenvectors) - i),
                       marker="s",
                       c=v)
  if title:
    plt.title(title)
  if filename:
    plt.savefig(filename)
  plt.close()

def plot_eigenvectors_table(data, results, eigenvectors, full=True, labels=None, title=None,
                      filename=None, offset_scale=0.1, azim=60, elev=30, proj_type='persp'):
  
  best_matches = results['best_matches']
  manifolds = results['manifolds']
  factors_one = manifolds[0][:4]
  factors_two = manifolds[1][:4]
  inv_map = {(v[0], v[1]) if v[0] in factors_one else (v[1], v[0]) : k for k, v in best_matches.items()}
  rows = 5
  cols = rows
  fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 2.25 * rows))
  fig.tight_layout()
  for r in range(rows):
    for c in range(cols):
      ax = axs[r, c] if len(eigenvectors) > 1 else axs
      index = r * cols + c
      if index >= len(eigenvectors):
        ax.set_visible(False)

      else:
        if r == 0 and c == 0:
          v = eigenvectors[0]

        elif r == 0:
          v = eigenvectors[factors_one[c-1]]

        elif c == 0:
          v = eigenvectors[factors_two[r-1]]
        
        else:
          v = eigenvectors[inv_map[(factors_one[r-1], factors_two[c-1])]]
        
        v /= np.linalg.norm(v)
        g = ax.scatter(data[:,0], data[:,1], marker="s", c=v)
        ax.axis('off')
        # fig.colorbar(g, ax=ax)
        ax.set_aspect('equal', 'datalim')
        # if labels is not None:
        #   ax.set_title(labels[index])

  if filename:
    plt.savefig(filename)
  plt.close()


def plot_independent_eigenvectors(manifold1, manifold2,
                                  n_eigenvectors,
                                  title=None, filename=None):
  '''
  plots a matrix with the independent eigenvectors color-coded by manifold
  '''

  # create separate manifolds
  data = np.zeros((1,n_eigenvectors))
  for i in manifold1:
    data[0,int(i)] = 1
  for j in manifold2:
    data[0,int(j)] = -1

  # create discrete colormap
  if 1 in manifold1:
    cmap = colors.ListedColormap(['blue', 'white', 'red'])
  else:
    cmap = colors.ListedColormap(['red', 'white', 'blue'])
  bounds = [-0.5,-0.25,0.25,0.5]
  norm = colors.BoundaryNorm(bounds, cmap.N)

  fig, ax = plt.subplots()
  ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')

  # draw gridlines
  ax.axes.get_yaxis().set_visible(False)
  ax.set_xticks(np.arange(0, n_eigenvectors, 5));

  if title:
    plt.title(title, fontsize=20)
  if filename:
    plt.savefig(filename)
  plt.close()

def plot_triplet_sims(sims, thresh=None, filename=None):
  '''
  plot the similarity scores of all triplets found
  '''
  counts, bins, patches = plt.hist(sims.values(), 50, density=True)
  if thresh:
    plt.plot([thresh, thresh], [0, np.max(counts)], "k--", linewidth=1)
    plt.annotate('thresh = {}'.format(thresh), xy=(thresh + 0.01, np.max(counts)))
  plt.yticks([])
  plt.title('Similarity score density')
  if filename:
    plt.savefig(filename)
  plt.close()

def plot_eigenmap(data_gt, phi, dims, filename=None):
  '''
  shows the laplace eigenmap of the data

  data_gt: the ground truth observations from the parameter space
  phi: the eigenvectors
  dims: the number of dimensions for the eigenmap
  '''
  if len(dims) < 2:
    print("Unable to create eigenmap: not enough eigenvectors")
    return
  fig = plt.figure()
  if len(dims) == 2:
    ax = fig.add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.scatter(phi[:,dims[0]], phi[:,dims[1]],
               s=5, c=data_gt, cmap='hsv')
  else:
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(phi[:,dims[0]], phi[:,dims[1]], phi[:,dims[2]],
               s=5, c=data_gt, cmap='hsv')
  if filename:
    plt.savefig(filename)
  plt.close()

def plot_product_sims(mixtures, phi, Sigma, steps, n_factors=2, filename=None):
  '''
  plots the similarity scores of all triplets for each product eigenvector

  mixtures: the complete list of product eigenvectors.
  phi: the complete matrix of eigenvectors.
  Sigma: the array of eigenvalues.
  steps: indexing to determine which product eigenvectors to plot.
  '''
  start, end, skip = steps
  num_plots = (end - start) // skip
  rows, cols = [2, int(np.ceil(num_plots / 2))]
  plt.rcParams.update({ "text.usetex": True, })
  fig, axs = plt.subplots(rows, cols, figsize=(1.5 * num_plots, 4), sharex=True)
  for index,eig_index in enumerate(mixtures[start:end:skip]):
    ax = axs[index // cols, index % cols]
    sims = []
    eigs = []
    v = phi[:,eig_index]
    lambda_v = Sigma[eig_index]
    for m in range(2, n_factors + 1):
      for combo in list(combinations(np.arange(1, eig_index), m)):
        combo = list(combo)
        lambda_sum = np.sum(Sigma[combo])
        lambda_diff = lambda_v - lambda_sum
        eigs.append(lambda_diff)

        # get product of proposed base eigenvectors
        v_combo = np.ones(phi.shape[0])
        for i in combo:
          v_combo *= phi[:,i]

        # test with positive
        sim = abs(cosine_similarity(v_combo, v))
        sims.append(sim)

    best_sim = np.max(sims)
    best_eig_err = eigs[np.argmax(sims)]
    ax.annotate('{}'.format(np.around(best_sim, 2)),
                 xy=(best_sim, best_eig_err),
                 xytext=(best_sim - 0.1, 0.75 * np.max(eigs)),
                 arrowprops=dict(arrowstyle='->',
                            color='r'))
    ax.set_title(eig_index)
    ax.set_xlim(0, 1)
    if index // cols == rows - 1:
      ax.get_xaxis().set_ticks([0, 0.5, 1.0])
    else:
      ax.get_xaxis().set_ticks([])
    if index % cols == 0:
      ax.set_ylabel(r'$\lambda_i + \lambda_j - \lambda_k$')
    ax.scatter(sims, eigs, s=3)
  plt.tight_layout(pad=0.5)
  if filename:
    plt.savefig(filename)
  plt.close()

def plot_C_matrix(manifolds, C, filename=None):
  '''
  plots a heat map of the separability matrix C.
  '''
  C1 = np.empty((C.shape[0], 0))
  idxs = []
  for m in manifolds:
    idxs.extend(m)
    C1 = np.concatenate((C1, C[:,m]), axis=1)

  C2 = np.empty((0, C1.shape[1]))
  for m in manifolds:
    C2 = np.concatenate((C2, C1[m,:]), axis=0)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_xticklabels(['']) # + idxs)
  ax.set_yticklabels(['']) # + idxs)
  matrix = ax.matshow(C2, cmap=plt.cm.Reds)
  fig.colorbar(matrix)
  if filename:
    plt.savefig(filename)
  plt.close()

def plot_fast_ica(data_transformed, data_gt, filename=None):
  '''
  plots the FastICA decomposition with color labels for the ground truth data
  note: currently only supports two factor decompositions
  '''
  fig = plt.figure()
  if data_transformed.shape[1] <= 2:
    plt.scatter(data_transformed[:,0], data_transformed[:,1], c=data_gt, cmap='hsv')
    plt.axis('off')
  else:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_transformed[:,0], data_transformed[:,1], data_transformed[:,2],
                c=data_gt, cmap='hsv')
    ax.axis('off')
  if filename:
    plt.savefig(filename)
  plt.close()

def plot_k_cut(labels,n_factors, theta, z):
  '''
  displays the max k-cut.
  '''
  plt.figure()
  for i in range(len(theta)):
    plt.plot([0, np.cos(theta[i])], [0, np.sin(theta[i])], c='red')
    plt.annotate(labels[0][i], xy=(np.cos(theta[i]), np.sin(theta[i])))
  for j in range(n_factors):
    z_angle = (z + j * 2 * np.pi / n_factors) % (2 * np.pi)
    plt.plot([0, np.cos(z_angle)], [0, np.sin(z_angle)], c='black')
  plt.xlim((-1.1, 1.1))
  plt.ylim((-1.1, 1.1))
  plt.axis('equal')
  plt.close()

def do_generate_plots(image_dir, info, result, eig_crit, sim_crit):
  name = info['name']
  datatype = info['datatype']
  phi = result['egvecs']
  Sigma = result['egvals']
  data = result['data']
  C = result['C_matrix']
  all_sims = result['all_sims']

  n_eigenvectors = Sigma.shape[0]

  if datatype == 'cryo-em':
    dimensions = [2 * info['x_stretch'], 2 * info['y_stretch'], 90]
    if name == 'cryo-em_x-theta_noisy':
      dimensions = [dimensions[0], dimensions[2], 0] # y = 0
  elif datatype == 'torus':
    dimensions = [2 * np.pi, 2 * np.pi, 0]
  else:
    dimensions = info['dimensions']

  if datatype == 'torus':
    data_dimensions = [2 * info['dimensions'][0],
                       2 * info['dimensions'][0],
                       2 * info['dimensions'][1]]
  else:
    data_dimensions = dimensions

  # get ground truth data to make plots clearer
  if datatype == 'cryo-em':
    image_data = info['image_data']
    data_gt = info['raw_data']
    if name == 'cryo-em_x-theta_noisy':
      data_gt = data_gt[:,[0,2]]
  else:
    data_gt = get_gt_data(data, datatype)

  print("\nSaving plots to", image_dir, flush=True)

  # plot original data
  print("Plotting original data...")
  if datatype == 'cryo-em':
    plot_cryo_em_data(image_data[:4],
                      filename='{}/{}'.format(image_dir, '{}_original_data.png'.format(name)))
  else:
    plot_synthetic_data(data, data_dimensions, azim=-30, elev=30,
                        filename='{}/{}'.format(image_dir, '{}_original_data.png'.format(name)))

  # plot eigenvectors
  vecs = [phi[:,i] for i in range(n_eigenvectors)]
  num_eigs_to_plot = 25
  print("Plotting first {} eigenvectors...".format(num_eigs_to_plot))
  eigenvectors_filename = '{}/{}_eigenvalues_{}.png'.format(image_dir, name, str(num_eigs_to_plot))
  plot_eigenvectors(data_gt, dimensions, vecs[:num_eigs_to_plot],
                    labels=[int(i) for i in range(num_eigs_to_plot)],
                    title='Laplace Eigenvectors',
                    filename=eigenvectors_filename)

  # plot manifolds
  manifolds = result['manifolds']
  independent_vecs = []
  for manifold in manifolds:
    vecs = [phi[:,int(i)] for i in manifold]
    independent_vecs.append(vecs)

  if datatype == 'rectangle3d':
    elev, azim = [30, -30]
  else:
    elev, azim = [30, 60]

  for m,vecs in enumerate(independent_vecs):
    print("Plotting eigenvectors for manifold {}...".format(m + 1))
    plot_eigenvectors(data_gt, dimensions, vecs[:5],
                      full=False,
                      labels=[int(j) for j in manifolds[m]],
                      filename='{}/{}'.format(image_dir, 'manifold{}_{}.png'.format(m, name)),
                      offset_scale=0,
                      elev=elev,
                      azim=azim)

  # plot 2d and 3d laplacian eigenmaps of each manifold
  for m in range(len(manifolds)):
    print("Plotting 2d laplacian eigenmap embedding for manifold {}...".format(m + 1))
    plot_eigenmap(data_gt[:,m], phi, manifolds[m][:min(2, len(manifolds[m]))],
                   filename='{}/{}'.format(image_dir, 'eigenmap{}_{}_2d.png'.format(m, name)))
    print("Plotting 3d laplacian eigenmap embedding for manifold {}...".format(m + 1))
    plot_eigenmap(data_gt[:,m], phi, manifolds[m][:min(3, len(manifolds[m]))],
                   filename='{}/{}'.format(image_dir, 'eigenmap{}_{}_3d.png'.format(m, name)))

  # plot sim-scores of best triplets
  print("Plotting all triplet similarity scores...")
  plot_triplet_sims(all_sims, thresh=sim_crit,
                    filename='{}/{}'.format(image_dir, 'triplet_sim-scores_{}.png'.format(name)))

  # plot mixture eigenvector sim-scores
  mixtures = get_product_eigs(manifolds, n_eigenvectors)
  steps = [5, 95, 15]
  print("Plotting select product eigenvector similarity scores...")
  plot_product_sims(mixtures, phi, Sigma, steps,
                    filename='{}/{}'.format(image_dir, 'product-eig_sim-scores_{}.png'.format(name)))

  # plot C matrix organized by manifold
  # print("Plotting separability matrix...")
  # plot_C_matrix(manifolds, C=C,
  #               filename='{}/{}'.format(image_dir, '{}_sep_matrix.png'.format(name)))