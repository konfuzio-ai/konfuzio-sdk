# simple docker file
FROM python:3.8-slim

ADD setup.py /code/setup.py
ADD konfuzio_sdk /code/konfuzio_sdk
ADD README.md /code/README.md

WORKDIR /code/

RUN python3.8 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" VIRTUAL_ENV="/opt/venv"

RUN pip install -e .
RUN pip install pytest