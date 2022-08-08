from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from datetime import timedelta
#from main import main

DeploymentSpec(name='train_model', tags=['tag1'], flow_name='main',
               schedule=IntervalSchedule(interval=timedelta(minutes=5)))
