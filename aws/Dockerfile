FROM python:3.9.8

WORKDIR /runner-app

ADD . /runner-app

COPY requirements.txt /runner-app/

RUN pip install -r requirements.txt

COPY . ./runner-app/

CMD ["python","./runner-app/tempsens_test_flask.py"]
