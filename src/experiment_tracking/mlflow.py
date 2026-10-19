
"""@package mlflow.py
Implementation of mlflow functionality.
"""
import numpy as np
import mlflow 
from src.utl.logging import Log as log
from src.experiment_tracking.exptrack_interface import ExperimentTrackingAPI as ET_API
from mlflow.models import infer_signature

class MLFlow(ET_API):
    def __init__(self, uri=None, experiment_name=None, verbose=False):
        super().__init__(uri=uri, experiment_name=experiment_name, verbose=verbose)
        self.experiment_id = None


    def init(self, run_id=None, run_name=None):

        if self.verbose:
            log.info(f"Connecting to server: {self.get_uri()}")
        mlflow.set_tracking_uri(self.uri)
        if self.verbose:
            log.ok()

        if self.verbose:
            log.info(f"Defining experiment name: {self.get_experiment_name()}")
        experiment = mlflow.set_experiment(self.experiment_name)
        self.experiment_id = experiment.experiment_id
        if self.verbose:
            log.ok()

        if run_id is None and run_name is None:
            returned_run_id = None
            returned_run_name = None

        else:
            # GS: if  not restart, create run by run_name
            # Refit: must never restart
            if run_id is None:
                returned_run_name = run_name
                if self.verbose:
                    log.info(f"Creating run by name: {returned_run_name}")
                    with mlflow.start_run(run_name=returned_run_name) as run:
                        returned_run_id = run.info.run_id
            else:
                returned_run_id = run_id
                runs = mlflow.search_runs(filter_string=f"attributes.run_id IN ('{returned_run_id}')", output_format='list')
                if len(runs) == 0:
                    log.err(f"Restart error! run id {returned_run_id} does not exist!")
                if len(runs) > 1:
                    log.err(f"Restart error! run id {returned_run_id} is associated to more than one run! ({len(runs)})")
                returned_run_name = runs[0].info.run_name
                if self.verbose:
                    log.info(f"Restart! Using already created run with id {returned_run_id}")
    
            if self.verbose:
                log.ok()
                log.info(f"Run id = {returned_run_id} -- run name = {returned_run_name}")

        return returned_run_id, returned_run_name
        
    def set_tag(self, tag=None, tag2=None):
        assert tag is not None, f"[ ERROR ] Tag must be defined!"
        mlflow.set_tag(tag, tag2)

    def log_array(self, param_name=None, value=None):
        assert param_name is not None, f"[ ERROR ] A param name must be defined!"
        assert value is not None, f"[ ERROR ] A value for this metric ({param_name}) must be defined!"
        assert isinstance(value, (list, tuple, np.ndarray)), f"value parameters must be a list, tuple or numpy array"
        mlflow.log_param(param_name, ",".join(map(str, value)))
    
    def log_param(self, param_name=None, value=None):
        assert param_name is not None, f"[ ERROR ] A param name must be defined!"
        assert value is not None, f"[ ERROR ] A value for this metric ({param_name}) must be defined!"
        mlflow.log_param(param_name, value)
    
    def log_params(self, params=None):
        assert params is not None, f"[ ERROR ] Params must be defined!"
        mlflow.log_params(params)
    
    def log_metric(self, metric_name=None, value=None):
        assert metric_name is not None, f"[ ERROR ] A metric name must be defined!"
        assert value is not None, f"[ ERROR ] A value for this metric ({metric_name}) must be defined!"
        mlflow.log_metric(metric_name, value)

    def log_image(self, image=None, key=None, step=None):
        mlflow.log_image(image, key=key, step=step)
    
    def log_figure(self, fig=None, key=None):
        mlflow.log_figure(fig, key)
    
    def log_model(self, model=None, artifact_path=None, inputs=None, predictions=None, conda_env=None):
        sig = None
        if inputs is not None and predictions is not None:
            sig = infer_signature(inputs, predictions)
        model_info = mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path, 
                                              signature=sig, #input_example=inputs, 
                                              conda_env=conda_env,
                                              await_registration_for=None, 
                                            )
        return model_info

    def get_experiment_id(self):
        return self.experiment_id

    def get_metric_history_by_run(self, run_id=None, metric=None):
        '''
        Fetch the metric history (returns a list of mlflow.entities.Metric objects).
        The history of an metric is ommited by search_runs for performance purposes.
        '''
        client = self.create_client()
        return client.get_metric_history(run_id=run_id, key=metric)

    def get_all_refit_runs(self, filter_substring='', experiment_ids=None):
        client = self.create_client()
        filter_string="attributes.run_name ILIKE 'REFIT_%" + filter_substring + "%'"
        if experiment_ids is None:
            runs = client.search_runs(self.experiment_id, filter_string=filter_string)
        else:
            runs = client.search_runs(experiment_ids, filter_string=filter_string)

        results = dict()
        for r in runs:
            results[r.info.run_name] = r
        return results

    def get_all_gs_runs(self, filter_substring=''):
        client = self.create_client()
        filter_string="attributes.run_name ILIKE 'GS_%" + filter_substring + "%'"
        runs = client.search_runs(self.experiment_id, filter_string=filter_string)
        results = dict()
        for r in runs:
            d = dict()
            d["run_obj"] = r
            filter_string=f"tags.`mlflow.parentRunId` = '{r.info.run_id}'"
            order_by=f"attributes.run_name ASC"
            #print(f"filter_string = {filter_string}")
            children = client.search_runs(self.experiment_id, filter_string=filter_string, order_by=[order_by])
            if len(children) > 0: 
                #print(f"Nmb of CHILDREN = {len(children)}")
                dc = dict()
                for c in children:
                    dc[c.info.run_name] = c
                d["children"] = dc
            else:
                #print(f"Nmb of CHILDREN = 0")
                d["children"] = None
            results[r.info.run_name] = d
        return results

    def get_runs_by_list_of_ids(self, ids_list=None, experiment_ids=None):
        assert ids_list is not None, f"[ INTERNAL ERROR ] ids_list is {ids_list}."
        client = self.create_client()
        str_list = ",".join([f"'{item}'" for item in ids_list])
        filter_string=f"attributes.run_id IN ({str_list})"
        if experiment_ids is None:
            runs = client.search_runs(self.experiment_id, filter_string=filter_string)
        else:
            runs = client.search_runs(experiment_ids, filter_string=filter_string)

        results = dict()
        for r in runs:
            results[r.info.run_name] = r
        return results
    
    def get_all_runs(self):
        client = self.create_client()
        runs = client.search_runs(self.experiment_id)
        return runs
    
    def get_experiments_by_prefix(self, prefix):
        client = self.create_client()
        experiments = client.search_experiments(filter_string=f"name LIKE '{prefix}%'")
        return experiments

    def create_client(self):
        return mlflow.MlflowClient(tracking_uri=self.uri)

    
#
#output_dir = Path("diabetes_artifacts")
#
#def log_data(
#    data, output_dir: Path, name: str
#) -> None:
#    data_dir = output_dir / "data"
#    if not data_dir.exists():
#        data_dir.mkdir(parents=True)
#    data_path = data_dir / f"{name}.npy"
#    np.save(data_path, data)
#    mlflow.log_artifact(data_path, "data")
#
#mlflow.sklearn.log_model(model, "model")
#log_data(X_test, output_dir, "x_test")
#log_data(y_test, output_dir, "y_test")
#log_data(X_train, output_dir, "x_train")
#log_data(y_train, output_dir, "y_train")





    
