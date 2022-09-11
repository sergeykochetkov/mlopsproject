FROM python:3.8-slim

WORKDIR /app

ADD cloud_function ./cloud_function
COPY ["model.py" ,"./cloud_function"]

RUN pip install -r cloud_function/requirements.txt

ENV PYTHONPATH=/app

EXPOSE 9696

CMD [ "python", "cloud_function/predict_service.py" ]