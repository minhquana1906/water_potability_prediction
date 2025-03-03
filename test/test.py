from pprint import pprint
from mlflow.tracking import MlflowClient
import mlflow
import dagshub

# Initialize MLflow with DagsHub
dagshub.init(repo_owner="minhquana1906", repo_name="water_potability_prediction", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/minhquana1906/water_potability_prediction.mlflow")
mlflow.set_experiment("Final model")

# client = MlflowClient()
# for mv in client.search_model_versions("name='sk-learn-random-forest-reg-model'"):
#     pprint(dict(mv), indent=4)

client = MlflowClient()
# for mv in client.search_model_versions("name='RandomForestClassifier'"):
#     pprint(dict(mv), indent=4)

# versions = client.get_model_version_by_alias("RandomForestClassifier", "staging")
# print(versions)
# # pprint(dict(versions), indent=4)
# load_model = mlflow.pyfunc.load_model(f"runs:/{versions.run_id}/{'RandomForestClassifier'}")

testmodel = client.search_model_versions("name='RandomForestClassifier'")[1]
print(testmodel)
