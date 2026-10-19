
"""@package local_data.py
Classes which represent models and data.
"""
import numpy as np

from src.utl.logging import Log as log

class ModelIdx:
    TRUE_MODEL_IDX = 0
    HIGHRES_MODEL_IDX = 1
    MEDIUMRES_MODEL_IDX = 2
    LOWRES_MODEL_IDX = 3
    WATER_MARK_MODEL_IDX = 4
    MAX_MODEL_IDX = 5

class DataIdx:
    HIGHRES_DATA_IDX = 0
    MEDIUMRES_DATA_IDX = 1
    LOWRES_DATA_IDX = 2
    MAX_DATA_IDX = 3



class Models2D:
    def __init__(self, models=None, verbose=False):
        self.m = models
    
    def get_models(self):
        return self.m
        
    def get_nx(self):
        return self.m.shape[2]
    
    def get_nz(self):
        return self.m.shape[1]
    
    def get_nmb_of_elems(self):
        return self.m.shape[0]
    
    def get_shape(self):
        return (self.m.shape[1], self.m.shape[2])

    def set_models(self, m):
        self.m = m
        

class DeepTomographyModels:

    def __init__(self, models=None, verbose=False):
        self.verbose = verbose
        self.models = models
    
    def get_true_models(self):
        return self.models[ModelIdx.TRUE_MODEL_IDX]
    
    def get_highres_models(self):
        return self.models[ModelIdx.HIGHRES_MODEL_IDX]
    
    def get_mediumres_models(self):
        return self.models[ModelIdx.MEDIUMRES_MODEL_IDX]
    
    def get_lowres_models(self):
        return self.models[ModelIdx.LOWRES_MODEL_IDX]
    
    def get_water_mask(self):
        return self.m[ModelIdx.WATER_MARK_MODEL_IDX]
    
    def get_model(self, idx):
        assert idx > 0 and idx < ModelIdx.MAX_MODEL_IDX, f"Invalid model index (idx = {idx}). Index must be in [0,{ModelIdx.MAX_MODEL_IDX-1}]"
        return self.m[idx]
    
    def get_nmb_of_elems(self):
        return self.models[ModelIdx.TRUE_MODEL_IDX].get_nmb_of_elems()


class SupervisedData:
    def __init__(self, X=None, Y=None, verbose=False):
        self.verbose = verbose
        self.X = X
        self.Y = Y
        if self.X is not None and self.Y is not None:
            assert self.X.shape[0] == self.Y.shape[0], f"[ ERROR ] X and Y must have the same number of elements!"
    
    def get_X(self):
        return self.X
    
    def get_Y(self):
        return self.Y
    
    def get_shape(self):
        return (self.X.shape[1], self.X.shape[2])
    
    def get_nmb_of_elems(self):
        return self.X.shape[0]

    def set_X(self, new):
        self.X = new
    
    def set_Y(self, new):
        self.Y = new
    

class Data:
    def __init__(self, data=None, current=DataIdx.HIGHRES_DATA_IDX, verbose=False):
        self.verbose = verbose
        self.data = data
        self.current = current

    def set_resolution(self, res):
        if res == 'high':
            self.current = DataIdx.HIGHRES_DATA_IDX
        elif res == 'med': 
            self.current = DataIdx.MEDIUMRES_DATA_IDX
        elif res == 'low': 
            self.current = DataIdx.LOWRES_DATA_IDX
        else:
            raise ValueError(f"Invalid resolutiont: {res}! Allowed resolutions are 'high', 'mid' and 'low'.")

    def get_current_resolution_name(self):
        ret = 'ERROR'
        if self.current == DataIdx.HIGHRES_DATA_IDX:
            ret = 'high'
        elif self.current == DataIdx.MEDIUMRES_DATA_IDX:
            ret = 'med'
        elif self.current == DataIdx.LOWRES_DATA_IDX:
            ret = 'low' 
        else:
            raise ValueError(f"Invalid resolutiont index: {self.current}! Allowed indexes are in [0,{DataIdx.MAX_DATA_IDX})")
        return ret

    def get_current_data(self):
        return self.data[self.current]
    
    def get_highres_data(self):
        return self.data[DataIdx.HIGHRES_DATA_IDX]
    
    def get_mediumres_data(self):
        return self.data[DataIdx.MEDIUMRES_DATA_IDX]
    
    def get_lowres_data(self):
        return self.data[DataIdx.LOWRES_DATA_IDX]
    
    def get_nmb_of_elems(self):
        if self.data[DataIdx.HIGHRES_DATA_IDX] is not None:
            ret =  self.data[DataIdx.HIGHRES_DATA_IDX].get_nmb_of_elems()
        elif self.data[DataIdx.MEDIUMRES_DATA_IDX] is not None:
            ret =  self.data[DataIdx.MEDIUMRES_DATA_IDX].get_nmb_of_elems()
        elif self.data[DataIdx.LOWRES_DATA_IDX] is not None:
            ret =  self.data[DataIdx.LOWRES_DATA_IDX].get_nmb_of_elems()
        else:
            ret = 0
        return ret 

