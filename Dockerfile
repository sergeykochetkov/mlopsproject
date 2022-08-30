FROM python:3.8-slim-buster

RUN pip install -U pip

RUN apt update
RUN apt install make wget -y

WORKDIR /app

COPY ["Makefile", "pre-commit", "requirements.txt", "./"]

RUN make install_anaconda
RUN make setup_cpu

COPY ["predict.py", "model.py", "./"]
COPY ["integration_test/mlruns/", "./mlruns"]


EXPOSE 9696

CMD ["bash", "-c", "source /root/anaconda3/etc/profile.d/conda.sh && conda activate MLOpsProject && gunicorn --bind=localhost:9696 predict:app"]