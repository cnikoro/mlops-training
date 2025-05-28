import pickle
import os
import mlflow

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    
    pwd = os.getcwd()

    path = os.path.join(pwd, 'models/')
    path_dv = os.path.join(path, 'dv.bin')
    #path_model = os.path.join(path, 'lin_reg.bin')
    
    #print(type(data[1]))
    # pickle dict vectorizer
    with open(path_dv, 'wb') as f_out:
        pickle.dump(data[0], f_out)

    ## pickle model
    #with open(path_model, 'wb') as f_out:
    #    pickle.dump(data[1], f_out)
  

    with mlflow.start_run():
        mlflow.log_artifact(local_path=path_dv, artifact_path="models_pickle")
        #mlflow.log_artifact(local_path=path_model, artifact_path="models_pickle")
        mlflow.sklearn.log_model(data[1], "lin_reg", signature=data[2])


