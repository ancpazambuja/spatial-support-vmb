
"""@package mlflow_skorch_callbacks.py
This file contains skorch callbacks to log in mlflow.
"""


import mlflow
from skorch.callbacks import Callback

from src.utl.logging import Log as log


#            mlflow.sklearn.log_model(gridsearch.best_estimator_, model_name, conda_env=None)

class ParamLogger(Callback):
    ''' Log training parameters.
    '''
    def on_train_begin(self, net, X, y):
        # Log all parameters to MLFlow
        for name, param in net.get_params().items():
            mlflow.log_param(name, param)

#class AppMLFlowLogger(MLflowLogger):
#    ''' Log training history and models.
#    '''







    
