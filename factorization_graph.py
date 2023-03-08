import numpy as np
import cvxpy as cp
import networkx as nx

from max_cut.max_cut_sdp import MaxCutSDP

def cosine_similarity(v_i, v_j):
  '''
  calculates the similarity score between vectors v_i and v_j
  S(v_i, v_j) = < v_i/||v_i||, v_j/||v_j|| >
  where S is the "similarity" function described in the paper
  '''

  # normalize vectors to unit norm
  v_i /= np.linalg.norm(v_i)
  v_j /= np.linalg.norm(v_j)

  sim = np.dot(v_i, v_j)
  return sim # score


def build_factorization_edges(best_matches: dict, best_sims: dict):
  edges = {}
  for prod, (i, j) in best_matches.items():
    if i == j: continue
    
    curr_sim = edges.get((i, j), 0)

    if best_sims[prod] >= curr_sim:
      edges[(i, j)] = best_sims[prod]

  return edges


def build_factorization_graph(best_matches: dict, best_sims: dict):
  graph = nx.Graph()
  nodes = set.union(
    {i for (i, j) in best_matches.values()},
    {j for (i, j) in best_matches.values()}
  )
  edges = build_factorization_edges(best_matches, best_sims)
  edges_nx = [(i, j, {'weight': w}) for (i, j), w in edges.items()]
  graph.add_nodes_from(nodes)
  graph.add_edges_from(edges_nx)
  return graph


def split_eigenvectors(best_matches: dict, best_sims: dict):
  '''
  Clusters eigenvectors into two separate groups
  returns:
  labels: a 2 x m array where m is the number of factor eigenvectors identified
          the first row of the array contains the indices of each factor eigenvector
          the second row of the array contains the label of the manifold factor
          to which the factor eigenvector corresponds
  C: the separability matrix
  '''
  graph = build_factorization_graph(best_matches, best_sims)
  sdp = MaxCutSDP(graph)
  sdp.solve()
  res = sdp._results
  cut = res['cut']
  labels = np.zeros((2, len(graph.nodes)))
  labels[0,:] = np.array(graph.nodes)
  labels[1, :] = np.array([max(cut[eigvec_index], 0) for eigvec_index in graph.nodes])

  return labels, res['matrix']
