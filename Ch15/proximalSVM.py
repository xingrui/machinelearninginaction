'''
Created on Feb 25, 2011

@author: Peter
'''
from numpy import *
import base64
import pickle

def map(key, value):
    # input key= class for one training example, e.g. "-1.0"
    classes = [float(item) for item in key.split(",")]   # e.g. [-1.0]
    D = diag(classes)
 
    # input value = feature vector for one training example, e.g. "3.0, 7.0, 2.0"
    featurematrix = [float(item) for item in value.split(",")]
    A = mat(featurematrix)
 
    # create matrix E and vector e
    e = ones((shape(A)[0],1))
    E = mat(append(A,-e,axis=1)) 
    print E
 
    # create a tuple with the values to be used by reducer
    # and encode it with base64 to avoid potential trouble with '\t' and '\n' used
    # as default separators in Hadoop Streaming
    producedvalue = base64.b64encode(pickle.dumps((dot(E.T, E), dot(dot(E.T, D), e))))    
 
    # note: a single constant key "producedkey" sends to only one reducer
    # somewhat "atypical" due to low degree of parallism on reducer side
    return "producedkey\t%s" % (producedvalue)
   
def reduce(key, values, mu=0.1):
  sumETE = None
  sumETDe = None
 
  # key isn't used, so ignoring it with _ (underscore).
  for _, value in values:
    # unpickle values
    ETE, ETDe = pickle.loads(base64.b64decode(value))
    if sumETE == None:
      # create the I/mu with correct dimensions
      sumETE = mat(eye(ETE.shape[1])/mu)
    sumETE += ETE
 
    if sumETDe == None:
      # create sumETDe with correct dimensions
      sumETDe = ETDe
    else:
      sumETDe += ETDe
 
    # note: omega = result[:-1] and gamma = result[-1]
    # but printing entire vector as output
    result = dot(linalg.inv(sumETE), sumETDe)
    print "%s\t%s" % (key, str(result.tolist()))

if __name__ == "__main__":
    output = map("-1.0", "3.0, 7.0, 2.0").split("\t")
    key, values = output
    reduce(key, [output])
