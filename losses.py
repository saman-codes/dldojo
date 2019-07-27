import numpy as np

class Loss():
    def output_gradient(self):
      return
  
class MSE(Loss):
  def __call__(self, predicted, labels):
    return 0.5 * np.square(predicted - labels)
  
  def output_gradient(self, predicted, labels):
    return predicted - labels
    
  
class CrossEntropy(Loss):
  def __call__(self, predicted, labels):
    return  -np.nan_to_num((labels*(np.log(predicted) + (1-labels)*(np.log(1-predicted)))))
  
  def output_gradient(self, predicted, labels):
    return predicted - labels 