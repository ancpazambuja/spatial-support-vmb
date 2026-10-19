
"""@package mlflow.py
Implementation of mlflow functionality.
"""
from src.utl.logging import Log as log
from src.data.data import SupervisedData, DeepTomographyModels
from src.visualization.data_viewer import DataAndModelsViewer as viewer
from src.experiment_tracking.mlflow import MLFlow
from src.experiment_tracking.exptrack_interface import ExperimentTrackingAPI as ET_API

import pandas as pd
import numpy as np

import tempfile
from datetime import datetime
import warnings
import os

#from mlflow import MlflowClient
import mlflow


import sklearn
from sklearn.model_selection import GridSearchCV




class MLFlowUtils:

    def create_client_and_get_run( uri=None, experiment_name=None, mlflow_=None, experiment_id=None, execution_id=None, execution_name=None):
        log.info(f"uri = {uri}")
        log.info(f"experiment_name={experiment_name}")
        log.info(f"experiment_id={experiment_id}")
        log.info(f"execution_id={execution_id}")
        log.info(f"execution_name={execution_name}")

        assert mlflow_ is not None or (mlflow_ is None and uri is not None and experiment_name is not None), f"parameters mlflow_ and (uri, experiment_name) are excludents."
        if mlflow_ is None:
            mlflow_ = MLFlow( uri=uri, experiment_name=experiment_name, verbose=True)
        client = mlflow_.create_client() # create a mlflow client for this thread
        run = MLFlowUtils.get_run(client=client, experiment_id=experiment_id, execution_id=execution_id)
        if execution_name is not None: 
            assert run.info.run_name == execution_name, f"Internal error! gvars' run_name and actual run_name differs!"
        return client, run

    '''
    Original code from https://gist.github.com/liorshk/9dfcb4a8e744fc15650cbd4c2b0955e5#file-mlflow_gridsearch-py
    '''

    def log_run(gridsearch:GridSearchCV=None, mlflow_client=None, experiment_id=None, 
        parent_run_id=None, subrun_name=None, run_index:int=None, tags=None,
        model_size=None, lr_scheduler=None, patience=None, seed=None,
        test_perc=None, train_indexes=None, test_indexes=None,
    ):
        """Logging of cross validation results to mlflow tracking server
        
        Args:
            run_name
            run_index (int): Index of the run (in Gridsearch)
        """
        
        cv_results = gridsearch.cv_results_
        nested_run = mlflow_client.create_run(experiment_id, run_name=subrun_name, tags={"mlflow.parentRunId": parent_run_id})
        #with mlflow.start_run(run_name=subrun_name, nested=True) as sub_run:  
        mlflow_client.set_tag(nested_run.info.run_id, "run_name", tags["run_name"]) # tags sao apenas informativas

        mlflow_client.log_param(nested_run.info.run_id,"folds", gridsearch.cv)
        mlflow_client.log_param(run_id=nested_run.info.run_id, key="model_size", value=model_size)
        mlflow_client.log_param(run_id=nested_run.info.run_id, key="lr_scheduler", value=lr_scheduler)
        mlflow_client.log_param(run_id=nested_run.info.run_id, key="patience", value=patience)
        mlflow_client.log_param(run_id=nested_run.info.run_id, key="_x_seed", value=seed)
        mlflow_client.log_param(run_id=nested_run.info.run_id, key="_x_test_perc", value=test_perc)
        mlflow_client.log_param(run_id=nested_run.info.run_id, key="_x_train_indexes", value=train_indexes)
        mlflow_client.log_param(run_id=nested_run.info.run_id, key="_x_test_indexes", value=test_indexes)

        log.info("Logging parameters")
        params = list()
        for p in gridsearch.param_grid:
            for k in p.keys():
               params.append(k)
        #params = list(gridsearch.param_grid.keys())
        for param in params:
            if param == 'callbacks__grad_clip__clip_value' and cv_results["param_%s" % param][run_index] is None:
                mlflow_client.log_param(nested_run.info.run_id, param, -1)
            else:
                mlflow_client.log_param(nested_run.info.run_id, param, cv_results["param_%s" % param][run_index])

        log.info("Logging metrics")
        for score_name in [score for score in cv_results if "mean_test" in score]:
            log.debug(f"\t logging '{score_name}' metric")

            score_name_aux = score_name.replace("test","valid")
            mlflow_client.log_metric(nested_run.info.run_id, score_name_aux, -cv_results[score_name][run_index])

            score_name_aux = score_name.replace("mean","std")
            score_name_aux = score_name_aux.replace("test","valid")
            mlflow_client.log_metric(nested_run.info.run_id, score_name_aux, cv_results[score_name.replace("mean","std")][run_index])

        log.info("Logging extra data related to the experiment")

        subrun_id = nested_run.info.run_uuid
        subrun_experiment_id = nested_run.info.experiment_id
        #mlflow.end_run()
        log.info(mlflow.get_artifact_uri())
        log.info("subrunID: %s" % subrun_id)
        log.info("experimentID: %s" % subrun_experiment_id)
    
    def log_results(gridsearch:GridSearchCV=None, mlflow_client=None, experiment_id=None, run=None, base_id=0, 
        model_size=None, lr_scheduler=None, patience=None, seed=None,
        test_perc=None, train_indexes=None, test_indexes=None,
    ): #model_name=None
        """Logging of cross validation results to mlflow tracking server
        """
        #mlflow_client.set_tag(run.info.run_id, "run_name", run.info.run_name) # obs.: a tag eh apenas informativa

#        if model_name is not None:
#            log.info(f"Logging model {model_name}")        
#            mlflow.sklearn.log_model(gridsearch.best_estimator_, model_name, conda_env=None) # gs refit must be True
#
#            log.info("Logging CV results matrix")
#            tempdir = tempfile.TemporaryDirectory().name
#            os.mkdir(tempdir)
#            timestamp = datetime.now().isoformat().split(".")[0].replace(":", ".")
#            filename = "%s-%s-cv_results.csv" % (model_name, timestamp)
#            csv = os.path.join(tempdir, filename)
#            with warnings.catch_warnings():
#                warnings.simplefilter("ignore")
#                pd.DataFrame(gridsearch.cv_results_).to_csv(csv, index=False)
#            mlflow.log_artifact(csv, "cv_results") 

        #mlflow.log_metric("best index", gridsearch.best_index_)
        #mlflow.log_metric("best - mean val score", -gridsearch.cv_results_['mean_test_score'][gridsearch.best_index_])
        #mlflow.log_metric("best - std val score", -gridsearch.cv_results_['std_test_score'][gridsearch.best_index_])
        for i, p in enumerate(gridsearch.cv_results_['params']):
            MLFlowUtils.log_run(gridsearch=gridsearch, mlflow_client=mlflow_client, experiment_id=experiment_id,
                parent_run_id=run.info.run_id, subrun_name=f"hpset_{base_id+i}", run_index=i, tags={"run_name":run.info.run_name},
                model_size=model_size, lr_scheduler=lr_scheduler, patience=patience, seed=seed,
                test_perc=test_perc, train_indexes=train_indexes, test_indexes=test_indexes,
            )
        #mlflow.end_run()

    def get_run(client=None, experiment_id=None, execution_id=None):
        run = client.search_runs(experiment_id, filter_string=f"attributes.run_id IN ('{execution_id}')")
        assert len(run) == 1, f"Internal error! There must be only one run returned for execution_id = {execution_id}"
        run = run[0]
        return run

    def get_metric_from_run(run=None, key=None):
        return run.data.metrics[key]

    def get_param_from_run(run=None, key=None):
        return run.data.params[key]

    def get_experiment_id_by_experiment_name(experiment_name=None):
        return mlflow.get_experiment_by_name(experiment_name).experiment_id

    def check_if_params_already_exist(mlflow_client=None, run=None, params=None):
        #log.info(f"params_dict = {params}")
        run_dict = run.data.to_dictionary()
        keys_already_exist = list()
        for k,v in params.items():
            if k in run_dict['params']:
                keys_already_exist.append(k)
        return len(keys_already_exist)>0, keys_already_exist

    def log_parameters_by_run(mlflow_client=None, run=None, params=None):
        run_id=run.info.run_id
        for k,v in params.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                mlflow_client.log_param(run_id=run_id, key=k, value=",".join(map(str, v)))
            else:
                mlflow_client.log_param(run_id=run_id, key=k, value=str(v))

    # Function to parse the string into a dictionary
    # Note: Code from bing AI.
    def string_to_dict(param_string):
        param_dict = {}
        lines = param_string.split('\n')
        current_key = None
    
        for line in lines:
            if line.strip() == '':
                continue
    
            if not line.startswith(' '):
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    if value == '!!python/tuple':
                        value = []
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        value = float(value)
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    param_dict[key] = value
                    current_key = key
                else:
                    key = line.split(':')[0]
                    param_dict[key] = {}
                    current_key = key
            else:
                if isinstance(param_dict[current_key], list):
                    param_dict[current_key].append(int(line.strip().lstrip('- ')))
                else:
                    sub_key_value = line.strip().split(': ', 1)
                    if len(sub_key_value) == 2:
                        sub_key, sub_value = sub_key_value
                        if sub_value.isdigit():
                            sub_value = int(sub_value)
                        elif sub_value.replace('.', '', 1).isdigit():
                            sub_value = float(sub_value)
                        elif sub_value.lower() == 'true':
                            sub_value = True
                        elif sub_value.lower() == 'false':
                            sub_value = False
                        param_dict[current_key][sub_key] = sub_value
    
        return param_dict


class MLFlowDataAndModelsLogger:
    '''
    Contains all the plot figures which will be logged in mlflow.

    Example:
    MLFlowDataAndModelsLogger dm_train_logger(indices, models, data)
    MLFlowDataAndModelsLogger dm_test_logger(indices, models, data)
    '''
    def __init__(self, verbose=True):
        self.verbose = verbose

        self.indices = None
        self.model_figs = None
        self.x_figs = None
        self.xy_figs = None
        self.figs_made = False
    
    def make_figs(self, indices, models:DeepTomographyModels, data:SupervisedData, colorbar=False, res_str=None):
        if models is not None:
            self.model_figs = viewer.model_view(models=models, samples=indices, colorbar=colorbar)
        self.x_figs = viewer.x_view(data=data.get_X(), samples=indices, colorbar=colorbar, title=res_str)
        self.xy_figs = viewer.xy_view(data=data, samples=indices, colorbar=colorbar, title=res_str)
        self.figs_made = True

    def log(self, mlflow_obj:MLFlow=None, title_prefix=""):
        assert mlflow_obj is not None, f"MLFlow object can't be None!"
        assert self.figs_made, f"You must call self.make_figs(...) method first!"
        
        if self.verbose:
            log.info("Logging models, x, y...") 

        if self.model_figs is not None:
            mlflow_obj.log_figure(self.model_figs, title_prefix+"_model_figs.png")
        mlflow_obj.log_figure(self.x_figs, title_prefix+"_x_figs.png")
        mlflow_obj.log_figure(self.xy_figs, title_prefix+"_xy_figs.png")
        
        if self.verbose:
            log.ok()

    def log_client(self, mlflow_client=None, run_id=None, title_prefix=""):
        assert self.figs_made, f"You must call self.make_figs(...) method first!"
        
        if self.verbose:
            log.info("Logging models, x, y...") 

        if self.model_figs is not None:
            mlflow_client.log_figure(run_id, self.model_figs, title_prefix+"_model_figs.png")
        mlflow_client.log_figure(run_id, self.x_figs, title_prefix+"_x_figs.png")
        mlflow_client.log_figure(run_id, self.xy_figs, title_prefix+"_xy_figs.png")
        
        if self.verbose:
            log.ok()


