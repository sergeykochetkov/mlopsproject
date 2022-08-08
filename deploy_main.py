from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

from main import main

DeploymentSpec(name='train_model', tags=['tag1'], flow=main,
               flow_runner=SubprocessFlowRunner(),
               schedule=IntervalSchedule(interval=timedelta(minutes=5)))
