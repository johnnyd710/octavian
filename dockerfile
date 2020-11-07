# python3.7-alpine3.8-2020-04-12
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6 as base

FROM base
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY checkpoint.pt /app/checkpoint.pt
COPY ./app /app