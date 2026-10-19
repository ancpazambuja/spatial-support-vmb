
"""@package exptrack_interface.py
Interface for experiment tracking systems.
"""

from src.utl.logging import Log as log


class ExperimentTrackingAPI:
    def __init__(self, uri=None, experiment_name=None, verbose=False):
        self.verbose = verbose
        self.uri=uri
        self.experiment_name=experiment_name
        assert self.uri is not None, f"[ ERROR ] A URI for MLFlow must be provided!"
        assert self.experiment_name is not None, f"[ ERROR ] A experiment_name  parameter must be provided!"
    
    def get_uri(self):
        return self.uri
        
    def get_experiment_name(self):
        return self.experiment_name

    def info(self):
        log.line()
        log.info(f"[ SERVER ] Uri: {self.uri}")
        log.info(f"[ PROJECT ] Experiment Name: {self.experiment_name}")
    
    
    def init(self):
        raise NotImplementedError
    
    def set_tag(self, tag=None, tag2=None):
        raise NotImplementedError

    def log_metric(self):
        raise NotImplementedError
    
    def log_param(self, param_name=None, value=None):
        raise NotImplementedError
    
    def log_params(self, params=None):
        raise NotImplementedError
    




    