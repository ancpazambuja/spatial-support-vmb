
"""@package local_serialization.py
This module contains helper classes for input/output.
"""
from glob import glob
import pandas as pd
import numpy as np
import joblib
import time


from sklearn import metrics
from sklearn import preprocessing

from src.utl.logging import Log as log
from src.utl.classes import ProjectDirs, DataInfo
from src.data.data import Models2D, DeepTomographyModels, SupervisedData, Data



class SerializationBase:
    """
    Base class for serialization (input / output).
    """
    def __init__(self, verbose=False):
        self.verbose = verbose


class Load(SerializationBase):
    """
    Base class for loading (reading) data. 
    """
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def load(self, filename=None):
        raise NotImplementedError
    
    
class Dump(SerializationBase):
    """
    Base class for dumping (writing) data. 
    """
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        
    def dump(self, data=None, filename=None):
        raise NotImplementedError

class CLoad(Load):
    """
    Load stream of bytes generate by write in C language.
    """
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def load(self, filename=None):
        start_time = time.time()
        if self.verbose:
            log.info(f"Loading data from <{filename}>")
        
        with open(filename, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
        end_time = time.time()
        if self.verbose:
            log.info(f"Time to load: {end_time - start_time}")
        return data

class NPLoad(Load):
    """
    Load with numpy package.
    """
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)

    def load(self, filename=None):
        start_time = time.time()
        if self.verbose:
            log.info(f"Loading data from <{filename}>")
        data = np.load(filename)
        end_time = time.time()
        if self.verbose:
            log.info(f"Time to load: {end_time - start_time}")
        return data


class PickleLoad(Load):
    """
    Load with pickle package.
    """
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
    
    def load(self, filename=None):
        start_time = time.time()
        if self.verbose:
            log.info(f"Loading data from <{filename}>")
        data = joblib.load(filename)
        end_time = time.time()
        if self.verbose:
            log.info(f"Time to load: {end_time - start_time}")
        return data



class NPDump(SerializationBase):
    """
    Base class for dumping (writing) data. 
    """
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        
    def dump(self, data=None, filename=None):
        start_time = time.time()
        if self.verbose:
            log.info(f"Dumping data to <{filename}>")
        np.dump(data, filename)
        end_time = time.time()
        if self.verbose:
            log.info(f"Time to dump: {end_time - start_time}")



class PickleDump(SerializationBase):
    """
    Base class for dumping (writing) data. 
    """
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        
    def dump(self, data=None, filename=None):
        start_time = time.time()
        if self.verbose:
            log.info(f"Dumping data to <{filename}>")
        joblib.dump(data, filename)
        end_time = time.time()
        if self.verbose:
            log.info(f"Time to dump: {end_time - start_time}")
