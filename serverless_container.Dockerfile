FROM python:3.8-slim-buster

WORKDIR /app

ADD cloud_function ./cloud_function
COPY ["model.py" ,"./cloud_function"]

RUN pip install -r cloud_function/requirements.txt

ENV PYTHONPATH=/app:/app/cloud_function


CMD [ "python", "-u", "cloud_function/predict_service.py" ]
#CMD ["gunicorn", "--bind=localhost:9696", "cloud_function.predict_service:app"]