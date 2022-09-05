FROM python:3.8-slim-buster

RUN pip install -U pip

RUN apt-get update
RUN apt-get install make wget -y

WORKDIR /app

COPY ["Makefile", "requirements.txt", "./"]

RUN make install_anaconda
RUN make setup_cpu


COPY ["lambda_function.py", "predict_service.py", "consts.py", "./"]
COPY ["mlruns/", "./mlruns"]

#CMD [ "source /root/anaconda3/etc/profile.d/conda.sh && conda activate MLOpsProject && python predict_service.py" ]
#ENTRYPOINT ['lambda_function.handler']