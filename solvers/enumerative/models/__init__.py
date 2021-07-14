MODEL_REGISTRY = {}
def RegisterModel(model_name):
    def decorator(m):
        MODEL_REGISTRY[model_name] = m
        return m

    return decorator

from models.uniform import *
#from models.bigram import *
#from models.ml_bow_unigram import *
from models.ml_bow_bigram import *
