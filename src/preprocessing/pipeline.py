"""@package local_pipeline.py
This module contains a series of customized transformations tailored for the preprocessing phase. 

It is based on sklearn Pipeline framework.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

import numpy as np

from src.utl.logging import Log as log
from src.data.data import DeepTomographyModels, Data

import torch

class BatchAbsMaxScaler(BaseEstimator, TransformerMixin):
    """
    Applys a normalization between [-1,1].
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.absmax = None
        self.name = "BatchAbsMaxScaler"

    def fit(self, X, y=None):
        v = self.verbose
        if v:
            log.info(f"Fitting: {self.name} calculating absmax!")

        abs = np.abs(X)
        self.absmax = np.max(abs, axis=(0,1))

        if v:
            log.ok()

        return self

    def transform(self, X, y=None):
        v = self.verbose
        if v:
            log.info(f"Transforming: applying {self.name}...")

        X = X/self.absmax

        if v:
            log.ok()

        return X

    def inverse_transform(self, X, y=None):
        X *= self.absmax
        return X



class BatchMinMaxScaler2(BaseEstimator, TransformerMixin):
    """
    Applys MinMaxScaler to a batch of 2d images alltogether.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.scaler = MinMaxScaler()
        self.original_shape = None

    #def fit_transform(self, X, y=None):
    #    log.info("Fit_transforming: BatchMinMaxScaler...")
    #    self.original_shape = X.shape
    #    X_flatten = X.reshape(-1,1)
    #    return self.scaler.fit_transform(X_flatten)

    def fit(self, X, y=None):
        v = self.verbose
        if v:
            log.info("Fitting: BatchMinMaxScaler stores the original shape!")

        self.original_shape = X.shape

        #X_flatten = X.ravel()   # raven returns a view of X. So it is a inplace transform.
        X_flatten = X.reshape(-1,1)
        print(f"X_flatten = {X_flatten.shape}")
        print(f"X_flatten.ndim = {X_flatten.ndim}")
        return self.scaler.fit(X_flatten, y)

    def transform(self, X, y=None):
        v = self.verbose
        if v:
            log.info("Transforming: applying batchminmaxscaler...")
            
        #X_flatten = X.ravel()   # ravel returns a view of X. So it is a inplace transform.
        X_flatten = X.reshape(-1,1)
        print(f"transf: X_flatten = {X_flatten.shape}")
        print(f"transf: X_flatten.ndim = {X_flatten.ndim}")
        X_flatten = self.scaler.transform(X_flatten, y)

        assert (X.shape == self.original_shape), f"[ ERROR ] Internal error! Shapes differ! {X.shape} != {self.original_shape}"

        if v:
            log.ok()

        return X


class ToTensor(BaseEstimator, TransformerMixin):
    """
    Convert from numpy to pytorch tensor.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X, y=None):
        v = self.verbose
        if v:
            log.info("Fitting: there is no need for fitting!")
        return self

    def transform(self, X, y=None):
        v = self.verbose
        if v:
            log.info("Transforming: converting from numpy to tensor...")
            
        X_tensor = torch.tensor(X, dtype=torch.float32)

        if v:
            log.ok()
        return X_tensor


class ToChannelsFirstOrder(BaseEstimator, TransformerMixin):
    """
    Convert from channels-last order (BHWC) to channels-first order (BCHW).
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X, y=None):
        v = self.verbose
        if v:
            log.info("Fitting: there is no need for fitting!")
        return self

    def transform(self, X, y=None):
        assert X.dim() == 3 or X.dim() == 4, f"Input tensor must have at least 3 dimensions (HWC or BHWC)."
        v = self.verbose
        if v:
            log.info("Transforming: from channels-last to channels first ordering...")

        if X.dim() == 3: 
            X = X.permute(2,0,1)
        else: 
            X = X.permute(0,3,1,2)

        if v:
            log.ok()

        return X.clone() # permute returns a view, so we are cloning

class XClipAndNormalization(BaseEstimator, TransformerMixin):
    """
    Makes a seimisc normalization in order to balance the amplitudes.
    To be applied just in seismic domain.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.vis_p_max = None
        self.vis_p_min = None
        self.clip = None
        self.n_channels = None

    def fit(self, X_train, y=None):
        v = self.verbose
        if v:
            log.info("Fitting: finding parameters for data normalization...")
            
        self.n_channels = X_train.shape[3]
        self.vis_p_max = np.percentile(X_train, 96, axis=(0,1,2))
        self.vis_p_min = -self.vis_p_max
        self.clip = (self.vis_p_max - self.vis_p_min) / 2.
        
        if v:
            log.ok()
            
        return self

    def transform(self, X, y=None):
        assert len(X.shape) == 4, f"X must have 4 dimensions! X.shape = {X.shape}"
        
        v = self.verbose
        if v:
            log.info("Transforming: normalization on seismic data...")
            
        np.clip(X[:,:,:,],-self.clip[:],self.clip[:],out=X)
        X[:,:,:,]=(X[:,:,:,]+self.clip[:])/(2*self.clip[:])
        
        if v:
            log.ok()
        return X


class YUnitTransform(BaseEstimator, TransformerMixin):
    """
    Transform from meters per seconds to kilometers per seconds.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, X_train, y=None):
        v = self.verbose
        if v:
            log.info("Fitting: there is no need for fitting!")
        #It is not necessary to calculate nothing. 
        return self

    def transform(self, X, y=None):
        assert len(X.shape) == 3, f"X must have 3 dimensions! X.shape = {X.shape}"
        v = self.verbose
        if v:
            log.info("Transforming: from meters per seconds to kilometers per second...")
        X[:] = X[:] / 1000
        if v:
            log.ok()
        return X


"""@name Pipeline Wrappers
@{
"""

class XPreprocessingPipe:
    def get_pipe(verbose=False):
        return Pipeline(
            [
                ( "XClipAndNormalization", XClipAndNormalization(verbose=verbose)),
                ( "ToTensor", ToTensor(verbose=verbose)),
                ( "ToChannelsFirstOrder", ToChannelsFirstOrder(verbose=verbose))
            ]
        )


class YPreprocessingPipe:
    def get_pipe(verbose=False):
        return Pipeline(
            [
                ( "YUnitTransform", YUnitTransform(verbose=verbose)),
                #( "BatchAbsMaxScaler", BatchAbsMaxScaler(verbose=verbose)),
                ( "ToTensor", ToTensor(verbose=verbose)),
            ]
        )

class ModelPreprocessingPipe:
    def get_pipe(verbose=False):
        return Pipeline(
            [
                ( "YUnitTransform", YUnitTransform(verbose=verbose)),
                ( "ToTensor", ToTensor(verbose=verbose)),
            ]
        )


class Preprocessing:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.x_pipe = XPreprocessingPipe.get_pipe(verbose=verbose)
        self.y_pipe = YPreprocessingPipe.get_pipe(verbose=verbose)
        self.m_pipe = ModelPreprocessingPipe.get_pipe(verbose=verbose)
    
    def __transform_X(self, data:Data=None):
        v = self.verbose

        if data.get_highres_data() is not None:        
            if v:
                log.printMessage(f"Preprocessing: transform X...")
                log.info(f"\tHigh resolution...")
    
            X = self.x_pipe.transform(data.get_highres_data().get_X())
            data.get_highres_data().set_X(X)
            
        if data.get_mediumres_data() is not None:        
            if v:
                log.info(f"Medium resolution...")
            X = self.x_pipe.transform(data.get_mediumres_data().get_X())
            data.get_mediumres_data().set_X(X)
            
        if data.get_lowres_data() is not None:        
            if v:
                log.info(f"Low resolution...")
            X = self.x_pipe.transform(data.get_lowres_data().get_X())
            data.get_lowres_data().set_X(X)
        
        if v:
            log.ok()
        
        return data
    
    def __transform_Y(self, data:Data=None):
        v = self.verbose
        
        if data.get_highres_data() is not None:        
            if v:
                log.printMessage(f"Preprocessing: transform Y...")
                log.info(f"\tHigh resolution...")
            Y = self.y_pipe.transform(data.get_highres_data().get_Y())
            data.get_highres_data().set_Y(Y)
            
        if data.get_mediumres_data() is not None:        
            if v:
                log.info(f"Medium resolution...")
            Y = self.y_pipe.transform(data.get_mediumres_data().get_Y())
            data.get_mediumres_data().set_Y(Y)
            
        if data.get_lowres_data() is not None:        
            if v:
                log.info(f"Low resolution...")
            Y = self.y_pipe.transform(data.get_lowres_data().get_Y())
            data.get_lowres_data().set_Y(Y)
        
        if v:
            log.ok()
        
        return data
    
    def __transform_Models(self, models:DeepTomographyModels=None):
        v = self.verbose
        
        if models.get_highres_models() is not None:        
            if v:
                log.printMessage(f"Preprocessing: transform Models...")
                log.info(f"\tHigh resolution...")
            m = self.m_pipe.transform(models.get_highres_models().get_models())
            models.get_highres_models().set_models(m)
            
        if models.get_mediumres_models() is not None:        
            if v:
                log.info(f"Medium resolution...")
            m = self.m_pipe.transform(models.get_mediumres_models().get_models())
            models.get_mediumres_models().set_models(m)
            
        if models.get_lowres_models() is not None:        
            if v:
                log.info(f"Low resolution...")
            m = self.m_pipe.transform(models.get_lowres_models().get_models())
            models.get_lowres_models().set_models(m)
        
        if models.get_true_models() is not None:        
            if v:
                log.info(f"True model...")
            m = self.m_pipe.transform(models.get_true_models().get_models())
            models.get_true_models().set_models(m)

        if v:
            log.ok()
        
        return models
    
    def transform(self, data:Data=None, models:DeepTomographyModels=None):
        self.__transform_X(data=data)
        self.__transform_Y(data=data)
        if models is not None:
            self.__transform_Models(models=models)
        return data, models
    
    def __fit_transform_X(self, data:Data=None):
        v = self.verbose
        
        if data.get_highres_data() is not None:        
            if v:
                log.info(f"Preprocessing: fit_transform X...")
                log.info(f"\tHigh resolution...")
            X = self.x_pipe.fit_transform(data.get_highres_data().get_X())
            data.get_highres_data().set_X(X)
            
        if data.get_mediumres_data() is not None:        
            if v:
                log.info(f"Medium resolution...")
            X = self.x_pipe.fit_transform(data.get_mediumres_data().get_X())
            data.get_mediumres_data().set_X(X)
            
        if data.get_lowres_data() is not None:        
            if v:
                log.info(f"Low resolution...")
            X = self.x_pipe.fit_transform(data.get_lowres_data().get_X())
            data.get_lowres_data().set_X(X)
        
        if v:
            log.ok()        
        
        return data
    
    def __fit_transform_Y(self, data:Data=None):
        v = self.verbose
        
        if data.get_highres_data() is not None:        
            if v:
                log.info(f"Preprocessing: fit_transform Y...")
                log.info(f"\tHigh resolution...")
            Y = self.y_pipe.fit_transform(data.get_highres_data().get_Y())
            data.get_highres_data().set_Y(Y)
            
        if data.get_mediumres_data() is not None:        
            if v:
                log.info(f"Medium resolution...")
            Y = self.y_pipe.fit_transform(data.get_mediumres_data().get_Y())
            data.get_mediumres_data().set_Y(Y)
            
        if data.get_lowres_data() is not None:        
            if v:
                log.info(f"Low resolution...")
            Y = self.y_pipe.fit_transform(data.get_lowres_data().get_Y())
            data.get_lowres_data().set_Y(Y)
        
        if v:
            log.ok()           
        
        return data
    
    def __fit_transform_Models(self, models:DeepTomographyModels=None):
        v = self.verbose
        
        if models.get_highres_models() is not None:        
            if v:
                log.info(f"Preprocessing: fit_transform Models...")
                log.info(f"\tHigh resolution...")
            m = self.m_pipe.fit_transform(models.get_highres_models().get_models())
            models.get_highres_models().set_models(m)
            
        if models.get_mediumres_models() is not None:        
            if v:
                log.info(f"Medium resolution...")
            m = self.m_pipe.fit_transform(models.get_mediumres_models().get_models())
            models.get_mediumres_models().set_models(m)
            
        if models.get_lowres_models() is not None:        
            if v:
                log.info(f"Low resolution...")
            m = self.m_pipe.fit_transform(models.get_lowres_models().get_models())
            models.get_lowres_models().set_models(m)
        
        if models.get_true_models() is not None:        
            if v:
                log.info(f"True models...")
            m = self.m_pipe.fit_transform(models.get_true_models().get_models())
            models.get_true_models().set_models(m)

        if v:
            log.ok()
        
        return models
    
    def fit_transform(self, data:Data=None, models:DeepTomographyModels=None):
        self.__fit_transform_X(data=data)
        self.__fit_transform_Y(data=data)
        if models is not None:
            self.__fit_transform_Models(models=models)
        return data, models