# use to run the algorithm on various datasets and generate figures
# for experiments.

import time
import os
import sys
import json
import pickle
import argparse

import numpy as np

from utils import *
from plots import *
from algorithm import factorize_manifold
from data.preprocessing import preprocess_cryo_em_data


def main():
  parser = argparse.ArgumentParser(description='Run experiments on geometric data.')
  parser.add_argument('--data', type=str, default=None, required=True, 
                      help='Path to pickle file containing information about the data.')
  parser.add_argument('--configs', type=str, default=None, required=True, 
                      help='Path to json file containing algorithm params.')
  parser.add_argument('--outdir', type=str, default=None, required=True, 
                      help='The directory in which to save the results.')
  parser.add_argument('--generate_plots', action='store_true', default=False,
                      help='Set to generate and save figures',
                      dest='generate_plots')
  arg = vars(parser.parse_args())

  # retrieve data
  info = pickle.load(open(arg['data'], "rb"))
  datatype = info['datatype']

  # load and unpack parameters
  with open(arg['configs']) as f:
    configs = json.load(f)

  print("\nParameters...")
  for item, amount in configs.items():
    print("{:15}:  {}".format(item, amount))

  sigma = configs['sigma']
  n_factors = configs['n_factors']
  n_eigenvectors = configs['n_eigenvectors']
  eig_crit = configs['eig_crit']
  sim_crit = configs['sim_crit']
  K = configs['K']
  seed = configs['seed']

  np.random.seed(seed)

  if datatype == 'cryo-em': # preprocess cryo-EM data
    image_data = info['image_data']
    image_data_ = preprocess_cryo_em_data(image_data)
    info['data'] = image_data_

  result = factorize_manifold(info['data'], sigma, n_eigenvectors, n_factors, eig_crit, sim_crit)
  print_manifolds(result['manifolds'])

  # create output directory
  if not os.path.exists(arg['outdir']):
    os.makedirs(arg['outdir'])
  result_file = '{}/{}'.format(arg['outdir'], 'results.pkl')
  print('\nSaving results to file', result_file, flush=True)

  # save info dictionary using pickle
  with open(result_file, 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")

  # generate plots
  if arg['generate_plots']:
    do_generate_plots(arg['outdir'], info, result, eig_crit, sim_crit)


if __name__ == "__main__":
  main()
