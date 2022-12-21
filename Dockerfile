# Dockerfile, Image, Container
FROM python:3.9.13

ADD models.py .

RUN pip install sklearn statsmodels fredapi python-bcb scipy darts pandas numpy azure-mgmt-resource azure-mgmt-datalake-store azure-datalake-store

CMD [ "python", "./models.py" ]

