# python3.7-alpine3.8-2020-04-12
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7 as base

FROM base AS train
COPY --from=requirements /app/requirements.txt .
RUN pip install -r requirements.txt
COPY ./app/model.py /app/model.py
RUN python3 /app/model.py

FROM base
COPY --from=requirements /app/requirements.txt .
RUN pip install -r requirements.txt
RUN ls
COPY --from=model /app/checkpoint.pt app/checkpoint.pt
COPY ./app /app