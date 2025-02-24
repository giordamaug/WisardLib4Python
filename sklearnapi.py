#
# WiSARD in python: 
# Classification and Regression
# by Maurizio Giordano (2022)
#
import numpy as np
import wisard
from utilities import *
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
mypowers = 2**np.arange(65, dtype = np.uint64)[::]

class WiSARDRegressor(BaseEstimator, RegressorMixin):
    """WiSARD Regressor """
    
    #def __init__(self,  nobits, size, map=-1, classes=[0,1], dblvl=0):
    def __init__(self,  n_bits=8, n_tics=256, random_state=0, code='t', debug=False):
        if (not isinstance(n_bits, int) or n_bits<1 or n_bits>64):
            raise Exception('number of bits must be an integer between 1 and 64')
        if (not isinstance(n_tics, int) or n_tics<1):
            raise Exception('number of bits must be an integer greater than 1')
        if (not isinstance(debug, bool)):
            raise Exception('debug flag must be a boolean')
        if (not isinstance(code, str)) or (not (code=='g' or code=='t' or code=='c')):
            raise Exception('code must either \"t\" (termometer) or \"g\" (graycode) or \"c\" (cursor)')
        if (not isinstance(random_state, int)) or random_state<0:
            raise Exception('random state must be an integer greater than 0')
        self._nobits = n_bits
        self._notics = n_tics
        self._code = code
        self._nrams = 0
        self._seed = random_state
        self._debug = debug
        self._nloc = mypowers[self._nobits]
        self._model = None
            
    def fit(self, X, y):
        self._retina_size = self._notics * len(X[0])   # set retina size (# feature x # of tics)
        self._nrams = int(self._retina_size/self._nobits) if self._retina_size % self._nobits == 0 else int(self._retina_size/self._nobits + 1)
        self._model = wisard.WiSARDreg(self._retina_size, self._nobits, map=self._seed)
        self._ranges = X.max(axis=0)-X.min(axis=0)
        self._offsets = X.min(axis=0)
        self._ranges[self._ranges == 0] = 1
        if self._debug: 
            timing_init()
            delta = 0                                   # initialize error
        for i,sample in enumerate(X):
            if self._debug:  print("Target %d"%y[i], end='')
            intuple = self._model._mk_tuple_float(sample, self._notics, self._offsets, self._ranges)
            self._model.train_tpl_val(intuple, y[i])        
            if self._debug:             
                res = self._model.response_tpl_val(intuple)
                delta += abs(y[i] - res)
                timing_update(i,y[i]==res,title='train ',size=len(X),error=delta/float(i+1))
        if self._debug: print()
        return self

    def predict(self,X):
        if self._debug: timing_init()
        y_pred = np.array([])
        for i,sample in enumerate(X):
            intuple = self._model._mk_tuple_float(sample, self._notics, self._offsets, self._ranges)
            y_pred = np.append(y_pred,[self._model.response_tpl_val(intuple)])
            if self._debug: 
                timing_update(i,True,title='test  ',clr=color.GREEN,size=len(X))
        if self._debug: print()
        return y_pred

    def __repr__(self): 
        return "WiSARDRegressor(n_tics: %d, n_bits:, %d, n_rams: %d, random_state: %d, n_locs: %r)\n"%(self._notics, self._nobits, self._nrams, self._seed,self._nloc)

    def __str__(self):
        ''' Printing function'''
        return "WiSARDRegressor(n_tics: %d, n_bits:, %d, n_rams: %d)\n"%(self._notics, self._nobits, self._nrams)

    def printRams(self):
        rep = ""
        for j in range(self._nrams):
            ep += f'{self._model._rams[j]}'
        return rep

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"n_bits": self._nobits, "n_tics": self._notics , "debug": self._debug, "code" : self._code, "random_state": self._seed
              }
    def getMapping(self):
        return self._model.getMapping()

    def getCode(self):
        return self._code

    def getNoBits(self):
        return self._nobits

    def getNoTics(self):
        return self._notics

    def getNoRams(self):
        return self._model.getNRams()

    def getClasses(self):
        return self._model.getClasses()
    
    def getRams(self):
        return self._model.getRams()


class WiSARDClassifier(BaseEstimator, ClassifierMixin):
    """WiSARD Regressor """
    
    def __init__(self,  n_bits=8, n_tics=256, random_state=0, code='t', 
            bleaching=True,default_bleaching=1,confidence_bleaching=0.01, debug=False):
        if (not isinstance(n_bits, int) or n_bits<1 or n_bits>64):
            raise Exception('number of bits must be an integer between 1 and 64')
        if (not isinstance(n_tics, int) or n_tics<1):
            raise Exception('number of bits must be an integer greater than 1')
        if (not isinstance(bleaching, bool)):
            raise Exception('bleaching flag must be a boolean')
        if (not isinstance(default_bleaching, int)) or n_bits<1:
            raise Exception('bleaching downstep must be an integer greater than 1')
        if (not isinstance(confidence_bleaching, float)) or confidence_bleaching<0 or confidence_bleaching>1:
            raise Exception('bleaching confidence must be a float between 0 and 1')
        if (not isinstance(debug, bool)):
            raise Exception('debug flag must be a boolean')
        if (not isinstance(code, str)) or (not (code=='g' or code=='t' or code=='c')):
            raise Exception('code must either \"t\" (termometer) or \"g\" (graycode) or \"c\" (cursor)')
        if (not isinstance(random_state, int)) or random_state<0:
            raise Exception('random state must be an integer greater than 0')
        self._nobits = n_bits
        self._notics = n_tics
        self._code = code
        self._nrams = 0
        self._nclasses = 0
        self._seed = random_state
        self._debug = debug
        self._nloc = mypowers[self._nobits]
        self._model = None
        #self._bleaching = bleaching
        #self._test = self.test_bleaching if self._bleaching else self.test
        #self._b_def = default_bleaching
        #self._conf_def = confidence_bleaching
        
    def fit(self, X, y):
        self._retina_size = self._notics * len(X[0])   # set retina size (# feature x # of tics)
        self._classes, y = np.unique(y, return_inverse=True)
        self._nclasses = len(self._classes)
        self._model = wisard.WiSARD(self._retina_size, self._nobits, self._classes, map=self._seed) 
        self._ranges = X.max(axis=0)-X.min(axis=0)
        self._offsets = X.min(axis=0)
        self._ranges[self._ranges == 0] = 1
        if self._debug: 
            timing_init()
            delta = 0                                   # initialize error
        for i,sample in enumerate(X):
            if self._debug:  print("Label %d"%y[i], end='')
            intuple = self._model._mk_tuple_float(sample, self._notics, self._offsets, self._ranges)
            self._model.train_tpl(intuple, y[i])        
            if self._debug: 
                res = self._model.test_tpl(intuple)
                delta += abs(y[i] - res)
                timing_update(i,y[i]==res,title='train ',size=len(X),error=delta/float(i+1))
        if self._debug: print()
        return self

    def predict(self,X):
        if self._debug: timing_init()
        y_pred = np.array([])
        for i,sample in enumerate(X):
            intuple = self._model._mk_tuple_float(sample, self._notics, self._offsets, self._ranges)
            y_pred = np.append(y_pred,[self._model.test_tpl(intuple)])
            if self._debug: timing_update(i,True,title='test  ',clr=color.GREEN,size=len(X))
        if self._debug: print()
        return y_pred

    def __repr__(self): 
        return "WiSARDClassifier(n_tics: %d, n_bits:, %d, random_state: %d, n_locs: %r)\n"%(self._notics, self._nobits, self._seed, self._nloc)

    def __str__(self):
        ''' Printing function'''
        return "WiSARDClassifier(n_tics: %d, n_bits:, %d)\n"%(self._notics, self._nobits)

    def printRams(self):
        rep = ""
        for cl in range(self._nclasses):
            rep += f'[{cl} '
            for j in range(self._nrams):
                rep += f'{self._model._layers.at(cl)[j]}'
            rep += '\n'
        return rep

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"n_bits": self._nobits, "n_tics": self._notics, 
            "debug": self._debug, "code" : self._code, "random_state": self._seed, 
            #"bleaching" : self._bleaching, "default_bleaching" : self._b_def  , "confidence_bleaching": self._conf_def
              }

    def getMapping(self):
        return self._model.getMapping()

    def getCode(self):
        return self._code

    def getNoBits(self):
        return self._nobits

    def getNoTics(self):
        return self._notics

    def getNoRams(self):
        return self._model.getNRams()

    def getClasses(self):
        return self._model.getClasses()
