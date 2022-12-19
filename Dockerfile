# Dockerfile, Image, Container
FROM python:3.9.15

ADD models.py .

RUN pip install sklearn statsmodels datetime fredapi bcb dateutil scipy darts pandas numpy

CMD [ "python", "./models.py" ]

