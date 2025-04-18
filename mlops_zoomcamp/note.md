# MLOps Zoomcamp Note

*MLOps = a set of best practice to putting Machine Learning to production*

General 3 stages of Machine Learning Project
1. Design: consider if machine learning is suitable to solve the problem
2. Train: Experiment Models
3. Operate: Deployment, Monitor model's performance

## Table of Contents
- [1.2 Environment Preparation](#12-environment-preparation)
- [1.3 (Optional) Training a ride duration prediction model](#13-optional-training-a-ride-duration-prediction-model)
- [1.4 Course Overview](#14-course-overview)
- [1.5 MLOps Maturity Model](#15-mlops-maturity-model)
- [2.1 Experiment Tracking Intro](#21-experiment-tracking-intro-mlflow)
- [2.2 Getting started with MLflow](#22-getting-started-with-mlflow)
- [2.3 Experiment tracking with MLflow](#23-experiment-tracking-with-mlflow)
- [2.4 Model Management](#24-model-management)
- [2.5 Model Registry](#25-model-registry)
- [2.6 MLflow in practice](#26-mlflow-in-practice)
- [2.7 MLflow Benefits, Limitations, and Alternatives](#27-mlflow-benefits-limitations-and-alternatives)
- [3.0.1 Machine Learning Pipeline](#301-machine-learning-pipeline)
- [3.0.2-3.5.x Setup Mage and Hands-on](#302-35x-setup-mage-and-hands-on)
- [4.1 Model Deployment Overview](#41-model-deployment-overview)
- [4.2 Web-services: Deploying models with Flask and Docker](#42-web-services-deploying-models-with-flask-and-docker)
- [4.3 Web-services: Getting the models from the model registry (MLflow)](#43-web-services-getting-the-models-from-the-model-registry-mlflow)
- [4.4 Streaming: Deploying models with Kinesis and Lambda](#44-streaming-deploying-models-with-kinesis-and-lambda)
- [4.5 Batch Preparing a scoring script](#45-batch-prepaing-a-scoring-script)
- [5.1 Intro to ML Monitoring](#51-intro-to-ml-monitoring)
- [5.2 Environment Setup](#52-environment-setup)
- [5.3 Prepare reference and model](#53-prepare-reference-and-model)
- [5.4 Evedently metrics calculation](#54-evedently-metrics-calculation)
- [5.5 Evidently Monitoring Dashboard](#55-evidently-monitoring-dashboard)
- [5.6 Dummy Monitoring](#56-dummy-monitoring)
- [5.7 Data Quality Monitoring](#57-data-quality-monitoring)
- [6. Best Practices](#6-best-practices)
- [6.1 Testing Python code with pytest](#61-testing-python-code-with-pytest)
    - [Pipenv](#pipenv)
    - [Pytest](#pytest)
- [6.2 Integration tests with docker-compose](#62-integration-tests-with-docker-compose)
- [6.3 Testing Cloud Services with LocalStack](#63-testing-cloud-services-with-localstack)
- [6.4 Code Quality: linting and formatting](#64-code-quality-linting-and-formatting)
    - [pyproject.toml](#pyprojecttoml)
- [6.5 Git pre-commit hooks](#65-git-pre-commit-hooks)
- [6.6 Makefile and make](#66-makefile-and-make)
- [6b.1 Terraform Introduction](#6b1-terraform-introduction)
- [6b.2 Terraform Modules and Outputs Variables](#6b2-terraform-modules-and-outputs-variables)

## 1.2 Environment Preparation

In the video, Alexey recommended to use linux operating system (EC2 in demonstration)
- Manually creating an instance on AWS cloud
- Creating "Key Pair", download it and place in ~/.ssh path
- specify instance's spec & launch
- Copying Public IP address then go to local terminal
    ```bash
    # username: ubuntu in demonstration
    ssh -i ~/.ssh/key_pair.extension <username>@<copied IP address>

    # connected to the remote instance
    ```
    - but we don't have to execute this bash command every time to connect to the remote instance, do this in local machine
    ```bash
    nano .ssh/config/
    
    # specify config
    Host <host-name>
        HostName <Public IP Address>
        User <Username>
        IdentityFile <Full Path to key-pair file location> 
        StrictHostKeyChecking no

    # then save and exit
    ssh <host-name>
    ```
- Installing py env (preparing python environment)
    ```bash
    wget <link to anaconda linux installer (.sh)>
    bash <downloaded .sh file>
    ```
- To open some service liek jupyter notebook, we might need to access via port such as "htpp://localhost:8888". We need to forward port before, which is very easy to do via vscode `PORT`.

## 1.3 (Optional) Training a ride duration prediction model
- Saving ML model
    ```python
    import pickle

    # from ... import model
    # model.fit(x_train, y_train)
    # dv = DictVectorizer().fit(...)

    with open("models/lin_reg.bin", "wb") as f_out:
        pickle.dump((dv, model), f_out)
    ```
- There're always DS' notebook where the model is developed is typically not well-arranged and not applicable in production.

## 1.4 Course overview
- Experiment Tracking and Model Management --> MLflow
- ML Pipeline --> Prefect & Kubeflow
    - Parameterized
- Model Deployment & Serving (used as a service)
- Monitoring and automatically Re-train model

## 1.5 MLOps Maturity Model
- Level-0: No MLOps 
    - No automation, All code in Jupyter Notebook (PoC)
- Level-1: DevOps, but No MLOps
    - What are included
        - Releases are automated
        - Unit & Integration Tests
        - CI/CD
        - OPs Metrics (Request workload & Network aspect)
    - What are not included
        - No Experiment Tracking
        - No Reproducibility
        - Data Scientists are separated from Engineers
- Level-2: Automated Training
    - Infrastructure
        - Trainning Pipeline
        - Experiment Tracking
        - Model Registry
    - Low Friction Deployment (maybe just bump version on platform)
    - DS works with Engineer Team
- Level-3: Automated Deployment
    - Easy to Deploy model
    - Full ML Pipeline: Data prep. -> Train Model -> Deploy model
    - A/b Testing
    - Model Monitoring
- Level-4: Full MLOps Automation
    - Automatic Train / Re-train / Deployment

## 2.1 Experiment Tracking Intro (MLflow)
- Install and how to use MLflow
- Important concept:
    - ML experiment: the process of building an ML model
    - Experiment run: each trial in an ML experiment
    - Run artifact: any file that is associated with an ML run
    - Experiment meta data
- Experiment tracking
    - Source code
    - Environment
    - Data
    - Model
    - Hyperparameters
    - Metrics
    - ...
    - For:
        - Reproducibility
        - Organization: finding information of this and that that we have developed before.
        - Optimization: tune to be a better ML model
- MLflow
    - open source platform for the machine learning lifecycle
    - A simple python package: `pip install mlflow`
    - components:
        - Tracking
        - Models
        - Model Registry
        - Projects
    - MLflow module allows you to organize your experiments into runs and keep track of
        - Parameters
        - Metrics
        - Metadata
        - Artifacts
        - Models
    
## 2.2 Getting started with MLflow
- To install dependencies that require for the project, you have to install it with `requirements.txt`, but the best practice is to use separated python environment such as conda or venv (virual env) to not mess up with packages installed in the local configuration.

```bash
conda create -n <environment-name> python=3.9

conda activate <environment-name>

# you will see something like (<environment-tracking-env> -> ...) in terminal
pip install -r requirements.txt

pip list

mlflow --version

# mlflow ui

# mlflow ui with backend database
mlflow ui --backend-store-uri sqlite:///mlflow.db

# then copy a given local ip address (and port) and open in the browser
```

*Note: **venv** vs **Conda***
| venv | conda |
|:-:|:-:|
| light weight | data-sci lib included as default|
| Pure Python Project | Accept Multi-language Project |

- To set mlflow tracker, go to notebook developing ML model
    ```python
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db") # it will try to save artifact to this db
    mlflow.set_experiment("custom-experiment-name")
    
    # develop your model in the notebook with mlflow lib
    with mlflow.start_run():
        
        mlflow.set_tag("developer", "dev_name") # this may be useful for a large team
        mlflow.log_params("train-data-path", "./path/to/data.parquet")
        mlflow.log_params("val-data-path", "./path/to/data.parquet")

        alpha = 0.1
        mlflow.log_params("alpha", alpha)
        
        lr = model(alpha)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
    ```
    - then go check experiment in mlflow UI 


## 2.3 Experiment tracking with MLflow
- Using tag is useful when you monitor the experiment and filter the model
- Using `compare` feature to visualize correlation of hyperparameters combination from search space that result the metric we specified.
- mlflow `autolog`

## 2.4 Model Management
- To avoid error prone by manually write it down somewhere or create confusing folders to contain different versions of model, use model logging from MLflow.
- log model
    1. `log artifact`, considering your model as another artifact from the run. It's not quite very useful.
        ```python
        # import mlflow
        import pickle

        # mlflow.set_tracking_uri("sqlite:///mlflow.db") # it will try to save artifact to this db
        # mlflow.set_experiment("custom-experiment-name")
        
        # develop your model in the notebook with mlflow lib
        # with mlflow.start_run():
            
            # mlflow.set_tag("developer", "dev_name") # this may be useful for a large team
            # mlflow.log_params("train-data-path", "./path/to/data.parquet")
            # mlflow.log_params("val-data-path", "./path/to/data.parquet")

            # alpha = 0.1
            # mlflow.log_params("alpha", alpha)
            
            # lr = model(alpha)
            # lr.fit(X_train, y_train)

            # y_pred = lr.predict(X_val)
            # rmse = mean_squared_error(y_val, y_pred, squared=False)
            # mlflow.log_metric("rmse", rmse)

            # save the model
            with open("models/lin_reg.bin", "wb") as f:
                pickle.dump(lr, f)
            f.close()

            mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="model_pickle/")
        ```
        - And now, you can see your local saved model in MLflow UI in "Artifacts" section. (we save the local saved model to MLflow *Artifacts*)
    2. `log model`, store more information about the model, and be later loaded easily.
        ```python
        # import mlflow

        # mlflow.set_tracking_uri("sqlite:///mlflow.db") # it will try to save artifact to this db
        # mlflow.set_experiment("custom-experiment-name")
        
        # develop your model in the notebook with mlflow lib
        # with mlflow.start_run():
            
            # mlflow.set_tag("developer", "dev_name") # this may be useful for a large team
            # mlflow.log_params("train-data-path", "./path/to/data.parquet")
            # mlflow.log_params("val-data-path", "./path/to/data.parquet")

            # alpha = 0.1
            # mlflow.log_params("alpha", alpha)
            best_params = {
                "learning_rate": 0.06,
                "max_depth": 30,
                "min_child_weight": 1.06,
                # etc
            }

            mlflow.log_params(best_params)
            
            # lr = model(alpha)
            # lr.fit(X_train, y_train)

            # y_pred = lr.predict(X_val)
            # rmse = mean_squared_error(y_val, y_pred, squared=False)
            # mlflow.log_metric("rmse", rmse)

            booster = xgb.XGBoostClassifier(...)
            mlflow.xgboost.log_model(booster, "models_mlflow")
        ```
        - The model is not xgboost, but the tutorial is. MLflow has multiple supported library to work with "log_model", check it out here: https://mlflow.org/docs/latest/models.html
        - Run, then check the UI, you will see the latest train model's detail is very well logged with parameters, metrics, saved model, and even project dependencies (as we specify in `conda` or `requirements.txt`)
        - In model folder detail, it will contain code snippet of how the saved model can be used or called from MLflow to predict on any new data.

## 2.5 Model Registry

Model Registry is a place in which all the models that are production-ready should be stored (it do not deploy anything).

Tracking Server (model run 1/2/3) >> ***Register Model*** >> Model Registry (staging/production/archive)

- promoting different model to the same model name, resulting in the new version of the model in the registry
- we can assign the stage for the models such as `Staging`, `Production`, `Archive` from `None` via mlflow UI

### MLflow Client
We are able to use MlflowClient class to interact with object in Mlflow besides WebUI
```python
from mlflow.tracking import MlflowClient

# instead of sqlite for local, this can be URL for remote server
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# request general detail
client.list_experiments()

client.create_experiment(name="custom-name-experiment")

# from mlflow.entities import ViewType
runs = client.search_runs(
    experiment_ids = '1'
    filter_string = "",
    # filter_string = "metrics.rmse < 6.8", # filtering
    run_view_type = ViewType
    max_results=5,
    order_by=["metrics.rmse ASC"]
)

for run in runs:
    print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['rmse']:.4f}")


# promoting to model registry
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# specify model performance condition
runs = client.search_runs(
    experiment_ids = '1'
    filter_string = "metrics.rmse < 6.8", # filtering
    run_view_type = ViewType
    max_results=5,
    order_by=["metrics.rmse ASC"]
)

# register the model
run_id = runs[0]
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(
    model_uri=model_uri,
    name="register-model-name"
)


# transitioning a model from one stage to another
client.list_register_models()

model_name = "register-model-name"
latest_versions = client.get_latest_versions(name=model_name)

# check stage
for version in latest_versions:
    print(f"version: {version.version}, stage: {version.current_stage}")

# transition the model
model_version = 4
new_stage = "Staging"
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=False
)

# annotate the model
client.update_model_version(
    name=model_name,
    version=model_version,
    description=f"The model version {model_version} was transitioned to {new_stage}"
)


# using components on Mlflow via Client in action!
# define utils function
def read_dataframe(file_name):
    df = pd.read_csv(filename)
    
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60>)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    
    return df

def preprocess(df, dv):
    # for ny-taxi dataset as shown in the video
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    train_dicts = df[categorical + numerical].to_dict(orient='records')
    return dv.transform(train_dicts)

def test_model(stage, X_test, y_test, name='register-model-name'):
    model = mlflow.pyfunc.load_model(f"models:/{name}/{stage}")
    y_pred = model.predict(X_test)
    return {"rmse": mean_squared_error(y_test, y_pred, squared=False)}

# load data
df = read_dataframe(filename)

# load fitted vectorizer to local
client.download_artifact(
    run_id = run_id,
    path='preprocessor',
    dst_path='.'
) # assume this will provide output path: 'preprocessor_path.b'

# assign preprocessor
with open("preprocessor_path.b", "rb") as f_in:
    dv = pickle.load(f_in)

X_test = preprocess(df, dv)

target = "duration"
y_test = df[target].values

# inference
score = test_model(name=model_name, stage="Staging", X_test=X_test, y_test=y_test)

print(score_dict)

# assume we're happy with the model in staging and want to transition to 'Production'
# transition the model
client.transition_model_version_stage(
    name=model_name,
    version=4,
    stage='Production',
    archive_existing_versions=True
)
```

In conclusion
The model registry component is a centralized mode lstore, set of APIs and a UI to collaboratively manage the full lifecycle of an MLflow Model.

It provides:
- Model lineage
- Model versioning
- Stage transitions
- Annotations

## 2.6 MLflow in practice
### Configuring MLflow
- Backend Store
    - Local Filesystem
    - SQLAlchemy Compatible DB (e.g. SQLite)
- Artifacts store
    - Local Filesystem
    - remote (e.g. s3 bucket)
- Tracking Server
    - no tracking server
    - localhost
    - remote

Consider these different scenarios to adopt MLflow:
1. A single data scientist working on an ML model.
    - Configurations
        - Backend Store: local filesystem
        - Artifacts Store: local filesystem
        - Tracking Server: no need
    - Remark
        - Even we don't enable tracking server, we are still able to use `mlflow ui`.
        - `mlflow ui` search the current working directory for *"mlrun"*. In case you re-structure your project, you must `cd` to the working directory that contains *mlrun* with a custom path, and then `mlflow ui` again.
        - Tips: In case you can't access `localhost` with access denied in web browser, just clear cookies and refresh the page.
2. cross-functional team working on an ML model (even with one data scientist to show the progress or tracking performance for the team).
    - Configurations
        - Backend Store: local.server
        - Artifacts Store: sqlite databse / Cloud
        - Tracking Server: localfile system
    - Remark:
        - You can run the tracking server locally by
            ```bash
            # mlflow server --backend-store-uri <backend-uri> --default-artifact-root <artifacts-path>
            mlflow server --backend-store-uri sqlite://backend.db --default-artifact-root ./artifacts_local
            ```
        - Setting `set_tracking_uri` would be localhost URL with a specific port
3. Multiple data scientists working on multiple ML models.
    - Configurations
        - Backend Store: postgres database (RDS)
        - Artifacts Store: Data Lake (S3 or GCS or blob storage)
        - Tracking Server: remote server (could be EC2)
    - Remark:
        - Setting up EC2 instance,
            ```bash
            sudo yum update
            pip3 install mlflow boto3 psycopg2-binary
            aws configure # For AWS authentication
            mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgres://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/DB_NAME --default-artifact-root s3://S3_BUCKET_NAME
            ```
            - go to URL: "http://<EC2_PUBLIC_DNS>:5000"
        - For AWS, you need "key-pair" to login through ssh into the instance as a credential.
        - Connection between RDS and EC2
            - In order to make EC2 can communicate with RDS, you need to add an inbound rule with PostgreSQL type, and the same security group with EC2's.
            - Copy RDS's endpoint and initialize session through EC2 with the MLflow command replaced backend-store-uri with created user, password and endpoint of the created RDS.
        - Use **Public IPv4 DNS** to connect to deployed MLflow.
        - `mlflow.get_tracking_uri()` is a good way to check if we specified a correct tracking server host's public DNS in setting uri part (`mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")`).
        - AWS resources:
            - instances/cluster: EC2
            - Data Lake: S3
            - Backend Database: RDS
For each run:
- Set Experiment
- Start run
- Log
    - Log Params
    - Log metric
    - Log model

## 2.7 MLflow Benefits, Limitations, and Alternatives
- Remote Tracking Server
    - Benefits
        - Sharing experiments with other data scientist
        - Collaborate with others (MLE, SRE) to build and deploy models
        - Give more visibility of the DS effort
    - Potential Issues
        - Security
            - Restrict access to the server (e.g. access through VPN)
        - Scalability
        - Isolation
            - All the artifacts will get messy quickly without stardard for naming experiment when working in multiple teams or tags, so it should be well-managed to make your artifact available to exploit.
            - All artifacts should be managed by different s3 bucket live on different account to avoid access restriction.
    - Limitations (and when not to use it)
        - Authentication & Users
            - The open source version of MLflow doesn't provide any sort of authentication
        - Data versioning
            - For the sake of reproducibility, data versioning is important, but MLflow doesn't provide build-in solution for the open source version.
        - Model/Data Monitoring & Alerting
            - This is out of scope of MLflow.
- MLflow alternatives
    - Neptune
    - Comet
    - Weights & Biases
    - and more

## 3.0.1 Machine Learning Pipeline
*A.K.A. Workflow Orchestration*
- A Sequence of steps to reproduce something (and for ML pipeline, reproducing an ML model)
- There're multiple tools can be mentioned here, some could be for general purpose, and some could be more specific to ML Pipeline.
    - More General
        - Airflow
        - Prefect
        - Mage
    - More specific to ML
        - Kubeflow pipelines
        - MLflow pipelines

## 3.0.2-3.5.x Setup Mage and Hands-on
TBD

## 4.1 Model Deployment Overview
- Design, Train (Experiment tracking + Training Pipeline), ***Operate***
- Types of deployment depends on requirement or how output would be used
    1. **Batch Deployment / Offline Deployment**
        - Model is not up and running all the time, we just apply our model to new data regularly (hourly, daily, monthly, etc.)
    2. **Online Deployment**
        - The model is always available, running up all the time
        - sub-types of online deployment
            1. Web Services (through HTTP request)
            2. Streaming (through stream of events)

### 1. Batch Deployment
- Database (historical data) -> ML services -> Database containing prediction
- Real-time Prediction is not necessary
- Example use case:
    - Marketing aspect, in term of churn prediction. We might not have to know immediately if the user is going to churn at the time, but we might need to predict whether the user is likely to churn and then we might give them some incentive after. It's just fine with daily batch or longer time interval for schedule.

### 2. Online Deployment
1. Web Service
    - The model needs to be up and running all the time, because user might want to decide action from prediction
    - Example use case:
        - Taxi App might have a backend service that communicates with ride duration prediction service. User may want to know ride duration, or how much it will cost immediately before decide to request the taxi from the app.
2. Streaming Service
    - consist of producer and consumer, producer push some event to an event stream and then consumer read from this stream and react to these events by doing something.
    - Backend now become the producer sending events to ML services which is independent consumers of that event stream.
        - One thing in the video that's interesting is web service and streaming service can be in the same system, but used in different purpose. For example, web service may handle the ride duration prediction first for user decision to request the taxi, but after user request, we might need more accurate model to infer ride duration for the user.  
    - The key difference between streaming service and web service is that streaming service (producer) pushs the event to the event stream and it doesn't really care what will be happened after, because producer and consumer don't have explicit connection between them ,unlike webservice which rely on API.
    - Example use cases:
        - Uploading Youtube video and searching for inappropriate content, "content moderation".
        - Recommendation services after the video upload to Youtube.
        ![mlops_streaming_service_example](./picture/mlops_streaming_service_example.png)

## 4.2 Web-services: Deploying models with Flask and Docker
1. Example of Flask app
    ```python
    # predict.py
    import pickle
    from flask import Flask, request, jsonify # request not requests


    with open("model/lin_reg.bin", "rb") as f_in:
        (vectorizer, model) = pickle.load(f_in)

    def pre_processing_input(input_features):
        features = {}
        # some feature engineering process, encoding, and transformation
        # features["feature_A"] = ...
        # features["feature_B"] = ...
        # features["feature_C"] = ...
        return features

    def predict(features):
        X = vectorizer.transform(features)
        y_pred = model.predict(X)
        return y_pred[0] # [0] for float (item in Sequence), since y_pred: numpy.array


    app = Flask("app-name")

    @app.route('/predict', method=['POST'])
    def predict_endpoint():
        input_features = request.get_json()

        features = pre_processing_input(input_features)
        y_pred = predict(features)

        result = {
            "target_feature": y_pred,
            # "model_version": RUN_ID
        }

        return jsonify(result)


    if __name__ = "__main__": # this block only used for local development (run locally)
        app.run(debug=True, host="0.0.0.0", port=9696)
    # for producttion, we use gunicorn
    ```
    - *Remark: it's a good practice to include model version it json prediction output*
    - gunicorn
        ```bash
        gunicorn --bind=0.0.0.0:9696 predict:app
        ```

2. Example of request test script
    ```python
    # test.py
    import requests


    URL = "http://localhost:9696/predict" # endpoint name
    input_features = {
        "feature_A": 10   
        "feature_B": 50
        "feature_C": "x"
    }

    response = requests.post(URL, json=input_features)
    print(response.json())
    ```

3. Example of Dockerfile for Flask app
    ```dockerfile
    FROM python:3.9.7-slim # light-weighted python version

    RUN pip install -U pip # some module required the latest pip v. e.g. xgboost
    RUN pip install pipenv # virtual py env

    WORKDIR /app

    COPY [ "Pipfile", "Pipfile.lock", "./" ]

    RUN pipenv install --system --deploy # --system to install directly to docker host, no need to isolate another python virtual environment within a container

    COPY [ "predict.py", "lin_reg.bin", "./" ]

    ENTRYPOINT [ "gunicorn", "--bind=localhost:9696", "predict:app" ]
    ```
- ***Remark***: This chapter often mentioned ML-Zoomcamp about how to productionize Docker Containerized application, such as using **Kubernetes**, or AWS Elastic Beanstalk. Maybe, it's a good idea to check it out:
    - [Machine Learning Zoomcamp 2022 Youtube Playlist](https://www.youtube.com/watch?v=MqI8vt3-cag&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)
    - [Chapter 5: Deploying Machine Learning Models](https://www.youtube.com/watch?v=agIFak9A3m8&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=49)
        - [Material: chapter-05-deployment](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-05-deployment)
    - [Chapter 10: Kubernetes and Tensorflow Serving](https://www.youtube.com/watch?v=mvPER7YfTkw&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=91)
        - [Material: chapter-09-kubernetes](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/chapter-09-kubernetes)

## 4.3 Web-services: Getting the models from the model registry (MLflow)

In the video, Alexey initialized MLflow server on a remote machine (EC2), and using s3 for artifact store and sqlite as a backend database.
```bash
mlflow server --backend-store-uri=sqlite://mlflow.db --default-artifact-root=s3://artifact-bucket/subpath
``` 

### Using Tracking server as a model registry

- In the notebook, DS might log the developed model and upload to the artifact bucket. And what we're gonna do is to retrieve the model from the model registry (the artifact bucket), instead of directly loading from local machine.
```python
# import pickle
# with open("lin_reg.bin", "rb") as f_in:
#     (dv, model) = pickle.load(f_in)

# see the example to load from mlflow web ui
import mlflow

# set connection to tracking server for retrieving the model
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# set properties
RUN_ID = "xxxx"
logged_model = f"runs:/{RUN_ID}/model"
model = mlflow.pyfunc.load_model(logged_model)

# we might need any other artifact, such as encoder, decoder, or vectorizer
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

ARTIFACT_PATH = client.download_artifacts(run_id=RUN_ID, path="dict_vertorizer.bin")
with open(ARTIFACT_PATH, "rb") as f_in:
    vertorizer = pickle.load(f_in)
```
- you might notice that loading vectorizer and model separately is quite a mess, we can clean this up by coupling the model and the vectorizer into a (sklearn) pipeline, and log it as a model. So, we can store this pipeline in the registry and don't need to store the artifact of vertorizer additionally.
```python
from sklearn.pipeline import make_pipeline

with mlflow.start_run():
    # params = {...}
    mlflow.log_params(params)

    pipeline = make_pipeline(
        dv = DictVectorizer(),
        model = RandomForestRegressor(**params)
    )

    # the comment line is an old version

    # X_train = dv.fit_transform(dict_train)
    # model.fit(X_train, y_train)
    pipeline.fit(dict_train, y_train)

    # X_val = dv.transform(dict_val)
    # y_pred = model.predict(X_val)
    y_pred = pipeline.predict(dict_val)


    rmse = mean_squared_error(y_pred, y_val, squared=False)
    mlflow.log_metric("rmse": rmse)

    # now, we log pipeline as a model 
    mlflow.sklearn.log_model(pipeline, artifact_path="model")
```

### Using data lake as a model registry
- Suppose, something could happen with the tracking server and cause it unavailable, we may not need our prediction service to get down too. So, we could use data lake as a model registry to fetch models from.
```python
# In MLflow UI model serving has full path of data lake storing model and artifact 

# we don't need to connect tracking server anymore, and use data lake path instead.
# MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# set properties
RUN_ID = "xxxx"
# logged_model = f"runs:/{RUN_ID}/model"
logged_model = f"s3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)
```

## 4.4 Streaming: Deploying models with Kinesis and Lambda

In this module, Alexey showed how to use AWS Lambda with Amazon Kinesis which required some role and permission, managaed in IAM, to run ML code on Lambda.

- Lambda allow us to run pieces of code in severless approach allowing us to less concerned with infrastructure management.

- Example Code on Lambda:
```python
import os
import json
import base64 # to decode message
import boto3

import mlflow

# use kinesis api to send each stream record to another stream event
kinesis_client = boto3.client("kinesis", region="ap-southeast-1")

def prepare_feature(input: dict) -> dict:
    preprocessed_features = {}
    return preprocessed_features

def predict(features: dict) -> float:
    # import model
    RUN_ID = os.getenv("MLFLOW_RUN_ID")
    logged_model = f"s3://bucket-name/1/{RUN_ID}/artifacts/model"
    model = mlflow.pyfunc.load_model(logged_model)
    
    # predict as float, unless error raised with JSON serialization issue, then return
    return float(model.predict(features))

def lambda_handler(event, context): # context is required in default config, but we don't have to worry about it. (can be provided as None when sending to this stream)

    # print(json.dumps(event)) # this will be used to observe payload event from Kinesis
    # return { # This is used for first test lambda
    #     "statusCode": 200,
    #     "body": json.dumps("Hello from Lambda!")
    # }

    # since event stream send event as list, we need loop and list
    predictions = []

    for record in event["Records"]:
        request_id = record["kinesis"]["id"] # get attribute according to payload
        raw_features = record["kinesis"]["data"] # event data might be encodedd with base64, required base64 or utf-8 decoding
        input_features = json.loads(raw_features)

        features = prepare_feature(input=input_features)
        prediction = predict(features)

        prediction_event = {
            "model": "some_model_name",
            "version": "123",
            "prediction": {
                "output_prediction": prediction,
                "id": request_id
            }
        }
        
        # sending to another stream
        kinesis_client.put_record(
            StreamName=os.getenv("STREAM_NAME", "default_value"),
            Data=json.dumps(prediction_event),
            PartitionKey=str(request_id),
        )

        predictions.append(prediction_event)

    return {"predictions": predictions} # this format in order to send to another stream or consumer
```

- Example of Kinesis
    - Configure every options in AWS Console Kinesis and create Data Stream
    - Use Kinesis as Triggerer in Lambda Service
    - Don't forget to attach permissions for Kinesis
    - Sending Data:
    ```bash
    aws kinesis put-record \
        --stream-name <stream-name> \
        --partition-key <ride-id> \
        --data '{
            "ride": {
                "input_feature_1": 130,
                "input_feature_2": 205,
                "input_feature_3": 3.66
            },
            "record_id": 156
        }'
    ```
    - Lambda now could read data from the stream (check logs)
    - Additionally, we could make use of shards to optimize the process if we have multiple shards for the stream

- Dockerfile
    - Another way to deploy code on Lambda is to use docker image and use *Container image* option to deploy Lambda function.
    - We might need additional dockerfile to achieve this, some of dependencies management, and ECR for image registry.
    - We have to concern how to authenticate the container with the services by exporting credentials as environment variable or attaching role to resoruces.

Conclusion
- We've learn deploy ML as a Lambda and let it consume input from stream event with Kinesis
- How Lambda works, How kinesis works
- How to put the model in the lambda
- How to put everything into docker and deploy it

## 4.5 Batch: Prepaing a scoring script

*Deploying model in offline mode (Batch)*

In this topic, Alexey demonstrate how to turn ML code from Jupyter Notebook to a clean py script which involve the following aspects:
- Extract only necessary transformation logic code
- Make it functional form (or OOP)
- Prepare I/O (Input Reader and Output Writer)
- Parameterize code
- Define input argument (should use `argparse`)
- Use Logging to indicate progress during execution

## 5.1 Intro to ML Monitoring

Most of time after deploying the ML to production stage, model performance will likely to degrade over time that might due to data drift, data outage, schema changes, concept drift, and etc. So, monitoring is the process to allow us to identify this kind of situation.

What should be monitoring are:
- Service health
- Model Performance
- Data quality and integrity
- Data and concept drift

Optionally, we should monitoring these too:
- Performance by segment
- Model bias and fairness
- Outliers
- Explainability

### Batch vs Online Serving Models
- How to monitor?
    - Additionall ddd ML metrics to service health monitoring tools e.g. Prometheus/Grafana
    - Build and ML-focused Dashboard(s) e.g. MongoDB/Grafana or BI Tools such as Tableau, Looker
- Example with Data Quality
    - Batch Models: Based on training data or past batch
        - Expected data quality e.g. 80% is non-constant and null
        - Data distribution e.g. normality
        - Descriptive Statistics e.g. avg, median, min-max, quantiles for individual feature
    - Non-batch Models
        - Calculate metrics continuously or even incrementally
        - Statistical test on a continuous data stream e.g. using window function and compare windows
- Monitoring Scheme
    - Software Service: request-response
    - I/O Logging: prediction logs
    - Monitoring Jobs: reference data vs ground truth
    - ML Evaluation Store: storing metrics with PostgresDB
    - Dashboard: Grafana to access stored metrics

## 5.2 Environment Setup

The topic is about configuring docker compose to set up environment for ML Monitoring using grafana

- docker-compose.yml
    ```yaml
    version: '3.7' # docker compose version

    networks:
        network-1:
        network-2:

    services:
        db:
            ...
            environment:
                POSTGRES_PASSWORD: password_example_1
            ports: 5432:5432
            networks:
                - network-1
        adminer:
            ...
        grafana:
            image: grafana/grafana
            user: "472"
            ports:
                - 3000:3000
            volume:
                - ./config/grafana_datasource.yaml:/etc/grafana/provisioning/datasources/datasource.yaml:ro
            networks:
                - network-1
                - network-2
            restart: always
    ```
- grafana config file: `grafana_datasource.yaml`
    ```yaml
    apiVersion: 1 # version of config file

    # list of datasources to insert/update
    # available in the database
    datasources:
        - name: PostgreSQL
        type: postgres
        access: proxy
        url: db.:5432
        database: test
        user: postgres
        secureJsonData:
            password: "password_example_1"
        jsonData:
            sslmode: "disable"
    ```

Run `docker compose up --build`, then access grafana UI via Web URL `http://localhost:3000`. Try to use default user in Grafana UI (admin-admin) and create new password.

## 5.3 Prepare reference and model

This chapter, instructor demonstate how to pre-processing data and create some simple model to predict result.

## 5.4 Evedently metrics calculation

This topic demonstate how to use eveidently to calculate different metrics related different aspects of machine learning pipeline by providing right argument for eveidently library. Then, it will visualize and give metrics that we specified that we need such as drift score, column drift, missing values, and etc.

- Example of evidently library usage
```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatsetDriftMetric, DatasetMissingValueMetric


column_mapping = ColumnMapping(
    target=None,
    prediction="prediction",
    numerical_feature=num_features,
    categorical_features=cat_features
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name="prediction"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ]
)

report.run(reference_date=train_data, current_data=val_data, column_mapping=column_mapping)

report.show(mode="inline")

result = report.as_dict() # observe result dict for metrics
```

## 5.5 Evidently Monitoring Dashboard

The topic, instructor showed how to create evidently monitoring dashboard which suit for batch model and the company that don't have monitoring infrastructure yet.

- Example

```python
from evidently.report import Report
from evidently.ui.workspace import Workspace


regular_report = Report(...)

ws = Workspace("workspace")

# project = ws.create_project("project name")
# project.description = "desc"
# project.save()

ws.add_report(project.id, regular_report)

# etc.
```

```bash
evidently ui
# then go to local host with shown port
```

## 5.6 Dummy Monitoring

This chapter, the instructor demonstrate how grafana work with postgres in practice. If we configure Grafana correctly, we can use Grafana to aggregate data in postgresDB and show as interactive dashboard on Grafana Web UI.

## 5.7 Data Quality Monitoring

This chapter, the instructor demonstrate how to integrate data quality metrics from eveidently to grafana dashboard which involve around inserting metrics to postgres database and use it to show on dashboard.

## 5.8 Save Grafana Dashboard

We can save developed Grafana Dashboard by storing configuration in a JSON file. Go to dashboard and try to find `Dashboard Settings` and save JSON Model section to a file. Mount that file to specific grafana's directory.

```yaml
services:
    grafana:
        ...
        volumes:
            - ./config/grafana_datasources.yaml:/etc/grafana/provisioning/datasources/datasources.yaml:ro
            - ./config/grafana_dashboard.yaml:/etc/grafana/provisioning/dashboards/dashboards.yaml:ro
            - ./dashboards:/opt/grafana/dashboards
```

## 5.9 Debugging with test suites and reports

The instructor demonstrate in this chapter of how we can utilize pre-built tests from evidently and integrated with machine learning pipeline, so we don't have to implement everything ourselves from scratch. Mostly the metrics is drift score and drifted columns related. These capabilities allow us to detect what's going on with our data, and we can then debug after being noticed. 

## 6. Best Practices

- Testing the code: unit test with pytest
- Integration test with LocalStack and Docker-Compose
- Code quality: linting, formatting, and pre-commit hooks
- Makefiles
- Development, staging and production environments (Theory)
- Infrastructure-as-code with Terraform and AWS (Implementing the environments)
    - Explanation of why we need to split across multiple environments
    - Demo: Terraform for Kinesis & Lambda, across multiple environments (staging and production)
- Automating deployments with CI/CD pipeline through GitHub Action
    - automatic deployment to staging and manual deployment with a click to prod env

## 6.1 Testing Python code with pytest

This chapter, Alexey showed how to setup `pipenv` on SSH, how to use pytest to test against all parts of kinesis and lambda code from chapter [4.4](#44-streaming-deploying-models-with-kinesis-and-lambda), and refactoring the code from the functional form to OOP to make it able to be tested.

And in the last part of the video, Alexey said that there's some function that we might don't want it to be executed during unit testing like putting record to message queue, so we have to adjust code a bit more to make it unavailable when unit testing is run.

### Pipenv

- `pipenv` is another python library that can be installed via `pip`. It solve dependency management issues by replacing traditional approach like python `virtualenv` or `requirements.txt` with `pipenv`.
- `pipenv` allow you to isolate your python environment for a python project, specify python version that you want, separate dependencies between environments (by `--dev`), and make you less concern about pinning all sub-dependencies by using `Pipfile` and `Pipfile.lock`.
- When an exact version isn’t specified in the Pipfile, the `pipenv` install command gives the opportunity for dependencies (and sub-dependencies) to update their versions to keep your dependencies up-to-date as most as possible.
- In order to distribute your python code as a package, you still need some additional tool like `setuptools`, `pyproject.toml`, or `wheel` to do so.
- Additionally, `pipenv` can leverage `.env` file at top-level of a repository to use in an isolated virtual environment opened by `pipenv shell`.

```bash
# install pipenv by pip
pip install pipenv
```

```bash
# enable pipenv
pipenv shell

# check which python is used
which python
pipenv --venv
pipenv --where
```
- Above command will automatically detect the current running virtual environment, and detect existing `requirement.txt` file to create `Pipefile` for the first time.

```bash
# Using [packages]
pipenv install
# pipenv install -r requirements.txt


# Using [packages] and [dev-packages]
pipenv install --dev
# pipenv install -r dev-requirements.txt --dev

# Remove dev dependencies
pipenv uninstall --all-dev
# pipenv uninstall --all

# Create/Update Pipfile.lock
pipenv lock

# Using Pipfile.lock
pipenv install --ignore-pipfile
```
- Typically, we shouldn't manually edit `Pipefile.lock` ourselves.
- Workflow should be something like below:
```bash
pipenv shell

# check if new added dependency compatible with the existing
pipenv install

pipenv lock

# usage in CI pipeline
pipenv install --dev

# usage in production
pipenv install --ignore-pipfile

pipenv run python <script-name>.py
```

*Remark: Don't forget that we can use environment variables that's specified in `.env` file at top level of the project in the isolated `pipenv` virtual environment*

Reference

- [Pipenv: A Guide to the New Python Packaging Tool](https://realpython.com/pipenv-guide/)

### Pytest

```
project_root
|
|-- tests
|   |-- __init__.py
|   |-- test_script.py
...
```

`test_script.py` should import the developed function that's gonna be used in production environment, then we can define `test_input` and `expected_output` within a function: `test_<function_name>` for `<function_name>` function.

```python
from custom.function.within.repo import multiply


def test_multiply():
    input = {...}
    expected_output = {...}

    output = multiply(input)

    assert output == expected_output
```

## 6.2 Integration tests with docker-compose

In this chapter, Alexey introduced how we can execute integration test to make sure if our code would operate correctly with other services. First, we need to pack our code with Docker to make it be able to be tested and deployed.

Alexey mostly showed how to refactor the code, and how it can be execute with docker commands.

## 6.3 Testing Cloud Services with LocalStack

This chapter Alexey demonstrated how we can simulate another service locally that our main code would interact with. For example, Alexey used `localstack/localstack` image with docker compose to spin up another service acted as Local Kinesis Service to enable integration test for main code and Kinesis in local machine.

Sometimes, we have to add/remove/refactor some part of code to make it able to be tested.

```yaml
services:
    backend:
        image: ${GET_FROM_INPUT_COMMAND}
        ports:
            - 8080:8080
        environment:
            SOME_VARIABLE_USED_IN_CODE_1: XXX
            SOME_VARIABLE_USED_IN_CODE_2: YYY
            SOME_VARIABLE_USED_IN_CODE_3: ZZZ
        volumes:
            - ./path/to/code:/app/
    kinesis:
        image: localstack/localstack
        ports:
            - 4566:4566
        environment:
            SERVICES: kinesis
```

```bash
# list streams in aws account
aws kinesis list-streams

# list streams in localstack
aws --endpoint-url=http://localhost:4566 kinesis list-streams

aws --endpoint-url=http://localhost:4566 kinesis create-stream \
    --stream-name <stream_name> \
    --shard-count 1
```

What is worth noting is that, Alexey mentioned that we can use `http://kinesis:4566/` to refer to `kinesis` service name which must be in the same network. If we have a service that's not in the same network, we need to connect via `localhost` instead of the specific service name. 

```bash
# running unit test
pipenv run pytest <directory_name>/

# running integration test
./integration_test/run.sh
```

```shell
# ./integration_test/run.sh

# Add shebang at the top
#!/usr/bin/env bash

# Start Testing Stack
docker compose up -d

sleep 5

aws --endpoint-url=http://localhost:4566 \
    kinesis create-stream \
    --stream-name ${STREAM_NAME} \
    --shard-count 1

# Run Test against testing Stack
pipenv run python test_docker.py

# Get Error code if exists
ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker compose logs
    docker compose down
    exit ${ERROR_CODE}
fi

# Run another test
pipenv run python test_kinesis.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker compose logs
    docker compose down
    exit ${ERROR_CODE}
fi

docker compose down
```

## 6.4 Code Quality: linting and formatting

Alexey introduced PEP8 which is a python community standard of how python should look like (Style and Guide for Python code) to improve code readibility and then considered it as a quality of code.

He also introduce one of many popular python linter tools: `pylint` to enable linting. Pylint do not only check the style of the code, but also check common mistakes written in the code that should be concerned.

```bash
pipenv install pylint --dev

pipenv run pylint <file_name>.py

# pipenv shell
# pylint --recursive=y .
```

However, it's possible to bypass some of linting rules for some reasons. We can use the below example to avoid linting checks for specific rule.

```python
class ClassName:

    # pylint: disable=too-few-public-methods
    
    def __init__(self):
        pass

    def some_function(self, unused_variable):
        # pylint: disable=unused-argument
        pass
```

Most of linter, we can customize its congiuration that's available such as `.pylintrc`, `ruff.toml` and etc.

There's some alternative including pylint can use different configuration file such as `pyproject.toml` so we don't have to have many different of configuration files in a project.

Example of `pyproject.toml`:
```toml
[tool.pylint.messages_control]

disable = [
    "missing-function-docstring",
    "missing-final-newline",
    "missing-class-docstring",
    "invalid-name"
]
```

There's also some other tools that are responsible for formatting, such as `black`, `isort`, etc. Formatter will try to change the content of the file to comply with convention/standards without changing how it work (refactoring, not optimizing). We also need these dependencies to be installed in development environment in order to use it.

All the commands are pretty simple and pre-built by the framework. You can find it easy to use by reading the official documentation.

Usage Example:

```bash
black -diff .

black ./path/to/script.py

isort --diff .

isort .
black .
pylint --recursive=y .
pytest tests/
```

### Pyproject.toml

*This tools is not directly introduced in the lesson, but somehow presented as a additional python project configuration file. However, I saw this configuration file quite often in many projects. So, I think it would be great to do some research on this an also considered this as good-to-know topic.*

`pyproject.toml` simplifies python project configuration written in `TOML` syntax and can be used with many other python ecosystem related tools for advanced usages.

`pyproject.toml` can help in the ***module search path*** relating to how you can import your developed program or code within a project regarding where you execute.
- When you execute python as a module; `python -m <directory>`, it will search for `__main__.py` in that `<directory>` and make all the script inside it able to be imported that relative to **the module directory**.
- `-m` flag add the script location to the module search path to make importing become consistent across the project.

*Note: Don't set `PYTHONPATH` manually, it can cause so much trouble. but the mentioned variable also relate to these kind of things*

```
snakesay-project/    ← The project root
│
├── snakesay/        ← The main module of this project
│   ├── __init__.py
│   ├── __main__.py  ← The entry point to snakesay
│   └── snake.py     ← The core of the program
│
├── .gitignore
├── LICENSE
├── pyproject.toml   ← What this tutorial is about
└── README.md
```

```python
# Example: __main__.py
import sys
from snakesay import snake

def main():
    snake.say(" ".join(sys.argv[1:]))

if __name__ == "__main__":
    main()
```

```bash
# Executed as a module
python -m snakesay <script-arguments> # This will work

# Executed as a script
python snakesay/__main__.py <script-argument> # This will not work due to importing failed
# Traceback (most recent call last):
#   ...
#   snakesay-project/snakesay/__main__.py", line 2, in <module>
#     from snakesay import snake
# ModuleNotFoundError: No module named 'snakesay'
```

By the way, `pyproject.toml` does not provide virtual environment. so we have to use some additional tools for this such as `venv`, `pipenv`, etc.

A minimal `pyproject.toml` configuration file could look like this:

```toml
[build-system]
requires = ["setuptools>=75.3.0"]
build-backend = "setuptools.build_meta"

[project]
name = "snakesay"
version = "1.0.0"

[project.scripts]
ssay = "snakesay.__main__:main"

[tool.setuptools.packages.find]
where = ["."]
```

- It require `setuptools` for installation.
- [tool.setuptools.packages.find] table tells your build-system, Setuptools, where to find packages in your project. In this case, it’s just the root directory. It can be useful when Setuptools can't find module in the project root directory.

Installing project setup from `pyproject.toml` to *virtualenv*:

```bash
$ (venv) python -m pip install -e .
```

- `-e` flag specify *editable*, making our changes in module can be used in real-time without re-installing
- However, if changes occur on `pyproject.toml` itself, we'll need to reinstall the packages.
- output file: `.egg-info` is metadata that should be hidden from version control.

We can also configure `pyproject.toml` to make our module become executable with alias such as:

```bash
ssay "<module-arguments>"
```

By modifying `pyproject.toml`

```toml
...

[project.scripts]
ssay = "snakesay.__main__:main"
# <module-alias> = "<module-name>.<main-script-name>:<entrypoint-function-name>"

...
```

- Please note that entrypoint function must not require any argument.

Futhermore, We can manage/pin dependencies for a module in `pyproject.toml` separately between multiple scenarios. Hence, we can replace `requirements.txt` and `requirements-dev.txt` with this approach.

```toml
...

[project]
name = "snakesay"
version = "1.0.0"
dependencies = ["rich"]

[project.optional-dependencies]
dev = ["black>=24.1.0", "isort"] # latest isort

...
```

*Note: `dev` is arbitrary name, it can also be some other name.*

For installing `dependencies` including `dev`:

```bash
python -m pip install -e ".[dev]"
```

However, we still can use `requirements.txt` if preferred:

```toml
...

[project]
name = "snakesay"
version = "1.0.0"
dynamic = ["dependencies", "optional-dependencies"]

[tool.setuptools.dynamic]
dependencies = { file = ["requirements.txt"] }
optional-dependencies.dev = { file = ["requirements-dev.txt"] }

...
```

Additionally, we can specify `__version__` variable in `__init__.py` to make package distribution version become dynamic, so we don't have to hardcode it in `pyproject.toml`.

```toml
...

[project]
name = "snakesay"
# version = "1.0.0"
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {attr = "snakesay.__version__"}
...
```

In the `__init__.py` script
```python
"""A CLI program that echoes a string with a bit of ASCII art."""

__version__ = "1.0.0"
```

*Note: notice that `dynamic` variable in `[project]` table in `pyproject.toml` refer to variable name within `pyproject.toml` that we can assign at runtime*

Moreover, some tools can leverage `pyproject.toml` to configure its usage. For example:

```toml
...

[tool.black]
line-length = 88

[tool.isort]
profile = "black"

...
```

Anyway, We still need some other tools to efficiently use `pyproject.toml` with such as `Poetry` and some other tools for distributing the module as a package with `build`, `twine`, `wheel` which are not covered in this topic.

For example of `pyproject.toml` for distribution capability:
```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "snakesay"
version = "1.0.0"
dependencies = ["rich"]
authors = [{name = "Jin Doe", email = "jindoe@example.com"}]
keywords = ["CLI", "ASCII Art"]
license = "MIT"
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

[project.optional-dependencies]
dev = ["black", "isort", "build", "twine"]

[project.scripts]
ssay = "snakesay.__main__:main"

[tool.setuptools.packages.find]
where = ["."]
```

*Note: we can also use `Private :: Do Not Upload` to prevent package from being uploaded.*

Example of how we can upload a package:
```bash
python -m build
# * Creating isolated environment: venv+pip...
# ...
# Successfully built snakesay-1.0.0.tar.gz and snakesay-1.0.0-py3-none-any.whl

# OR
python -m twine upload dist/*
# This command will upload all the distributions in the dist directory to PyPI
```

Here is full example of `pyproject.toml` could look like:

```toml
[build-system]
requires = ["setuptools>=75.3.0"]
build-backend = "setuptools.build_meta"

[project]
name = "snakesay"
dependencies = ["rich>=13.9.0"]
authors = [{name = "Jin Doe", email = "jindoe@example.com"}]
keywords = ["CLI", "ASCII Art"]
readme = {file = "README.md", content-type = "text/markdown"}
requires-python = ">=3.9"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
dynamic = ["version"]

[project.urls]
Repository = "https://github.com/me/spam.git"
Issues = "https://github.com/me/spam/issues"

[project.optional-dependencies]
dev = ["black>=24.1.0", "isort>=5.13.0", "build", "twine"]

[project.scripts]
ssay = "snakesay.__main__:main"

[tool.setuptools.packages.find]
where = ["."]

[tool.setuptools.dynamic]
version = {attr = "snakesay.__version__"}

[tool.black]
line-length = 88

[tool.isort]
profile = "black"
```

| Tool | Manages dependencies | Virtualenv | Build metadata | Standard |
|:----:|:--------------------:|:----------:|:--------------:|:--------:|
pipenv | ✅ | ✅ | ❌ | ❌
pyproject.toml | ✅ (via tools like poetry) | ❌ (external) | ✅ | ✅ (PEP 518+)

Reference

- [How to Manage Python Projects With pyproject.toml](https://realpython.com/python-pyproject-toml/)

## 6.5 Git pre-commit hooks

The pre-commit hook is run first, before you even type in a commit message in order to regulate some restrictions to the code. We also need to install it first before usage.

```bash
pipenv install --dev pre-commit

pre-commit install

# .pre-commit-config.yaml
```

We can check the capability of pre-commit hook, like what it can do and cannot by observing available features in pre-commit hook library. There're some interesting hooks to mention such as `check-yaml`, `detect-private-key`, etc.

*Note: we need `.git` folder and its contents to enable pre-commit hook*

## 6.6 Makefile and make

```bash
# for Windows
choco install make

# don't forget to restart vscode to make it available after installation
```

With Makefile and `make` operation, we can execute a set of bash operations as alias make command.

```makefile
.SILENT: test lint test integration-test start stop build publish

IMAGE_TAG:=$(shell date + "%Y%m-%d %H-%M-%S")

LOCAL_IMAGE_NAME:=some-image-name:${IMAGE_TAG}
LOCAL_IMAGE_NAME="image-name-on-repo:${IMAGE_TAG}"

lint:
    isort .
    black .
    pylint --recursive=y .

test:
    pytest . PYTHONPATH=tests/

integration-test:
    bash ./integration_test.sh

start:
    docker compose -f ./docker/Dockerfile up --build

stop:
    docker compose -f ./docker/Dockerfile up down --rm

build: test integration-test
    echo "Building Image: ${LOCAL_IMAGE_NAME}"
    docker build -t ${LOCAL_IMAGE_NAME} .

publish: build
    docker tag ${LOCAL_IMAGE_NAME} ${REPO_NAME}
    docker push

```

```bash
make lint

make build
```

## 6b.1 Terraform Introduction

This chapter, instructor describe in overview of what Terraform is, and introduce basic lessons of Terraform usage from Data Engineering Zoomcamp.

**Terraform** is Infrastructure as Code (IaC) tool to allow us manage cloud resource with code which is consider as a best practice by keep our code versioned in version control. 

## 6b.2 Terraform Modules and Outputs Variables

Modules are used to create reusable components for Terraform which is very useful to avoid code duplication for multiple environments such as staging and production. 

*In-progress . . .*
