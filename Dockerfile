FROM tensorflow/tensorflow:1.15.0-py3

USER root

WORKDIR /app

COPY nst.py /app/
COPY nst_utils.py /app/
COPY requirements.txt /app/
COPY imagenet-vgg-verydeep-19.mat /app/

RUN pip install --upgrade pip && pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python", "-u", "nst.py"]
