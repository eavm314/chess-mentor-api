
FROM python:3.11-slim


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./src /code/src


CMD ["fastapi", "run", "src/app.py", "--port", "8000"]