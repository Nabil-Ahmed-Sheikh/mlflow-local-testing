from hashlib import new
import os
import mlflow
from mlflow import log_metric, log_param, log_artifact
from  mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://a279a37908c9343e58d473fc0bf3e85c-335940171.us-west-2.elb.amazonaws.com:5000")
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

def create_experiment(name, artifact_location, tags):
    client = MlflowClient()
    return client.create_experiment(name, artifact_location, tags)

def add_experiment_tag(experiment_id, key, value):
    client = MlflowClient()
    return client.set_experiment_tag(experiment_id, key, value)

if __name__ == "__main__":

    # create_experiment("test1x", "gg.com", [])
    MlflowClient().create_run(experiment_id="1")
   
   # Log a parameter (key-value pair)
    log_param("param1", 5)
    
    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", 1)
    log_metric("foo", 2)
    log_metric("foo", 3)


    # Log an artifact (output file)
    with open("output2.txt", "w") as f:
       f.write("Hello world!")
    
    file = open("output2.txt","r") 
    print (file.read()) 

    log_artifact("output2.txt")

   
    print(mlflow.get_artifact_uri())
