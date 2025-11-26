
import sys
sys.path.append('/content/drive/MyDrive/myproject')
import numpy as np
from common.layers import Embedding
from ch4.negative_sampling_layer import NegativeSampliingLoss

class CBOW:
  def __init__(self, vocab_size, hidden_size,window_size,corpus):
    V,H = vocab_size, hidden_size

    W_in = 0.01* np.random.randn(V,H).astype('f')
    W_out = 0.01 * np.random.randn(V,H).astype('f')

    self.in_layers = []
    for i in range(2*window_size):
      layer = Embedding(W_in)
      self.in_layers.append(layer)
    self.neg_sampling_loss = NegativeSampliingLoss(W_out,corpus,0.75,5)

    layers = self.in_layers + [self.neg_sampling_loss]
    self.params, self.grads = [],[]
    for layer in layers:
      self.params = layer.params
      self.grads = layer.grads
    
    self.word_vecs = W_in

  def forward(self, contexts, target):
    h=0
    for i,layer in enumerate(self.in_layers):
      h += layer.forward(contexts[:,i])
    h *= 1/len(self.in_layers)
    loss = self.neg_sampling_loss.forward(h, target)
    return loss
  
  def backward(self, dout=1):
    d_h = self.neg_sampling_loss.backward(dout)
    d_h *= 1/len(self.in_layers)
    for layer in self.in_layers:
      layer.backward(d_h)
    return None
