The API directory contains the files to deploy locally the ML-model.

The model used is found in API/app/model/model_day_AP001-0.1.0.pkl

It corresponds to a Linear Regression that takes the AQI-score of
previous 20 days to predict the sucessive day AQI-score. Check 
API/app/model/sample.json to see the general input structure to 
make predictions.

The API directory contains three bash scripts:
* restart_docker.sh use in case ports are blocked
* 0_docker_build.sh use to build the docker project
* 1_run_container.sh use to launch locally the project

The rest of the files in the API directory are files needed to dockerize 
the project.

The API/app/model/model.py is used to load the project and hardcode
the version.

The API/main.py file contains the general code of the app.

Input: 20 previous measurements
Output: AQI-score and air-quality prediction.

Go to http://0.0.0.0/docs to test the app.

