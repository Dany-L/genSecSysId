import os

# The tests use MLflow's file store, which newer MLflow versions only allow with this opt-in.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
