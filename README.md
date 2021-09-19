# MLFlow example
Data in docker container example (normally do not do it this way, mount it instead!)
```
docker build -t mnist-model -f docker_data/.Dockerfile docker_data/
```

First run the MLFlow UI
```
mlflow ui
```

Running the MLproject driver
```
python mlflow_project_driver.py
```

Check MLFlow UI for metrics.