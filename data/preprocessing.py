# Code taken from the original implementation: https://github.com/sxzhang25/product-manifold-learning

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocess_cryo_em_data(image_data, random_state=0):
  n_samples = image_data.shape[0]

  # apply PCA and standard scaling to the data
  print("\nApplying PCA and standard scaling...")
  image_data_ = np.reshape(image_data, (n_samples, -1))
  pca = PCA(n_components=4, random_state=random_state)
  image_data_ = pca.fit_transform(image_data_)

  # standard scale each channel
  scaler = StandardScaler()
  image_data_ = np.reshape(image_data_, (-1, image_data_.shape[1]))
  image_data_ = scaler.fit_transform(image_data_)
  image_data_ = np.reshape(image_data_, (n_samples, -1))

  return image_data_