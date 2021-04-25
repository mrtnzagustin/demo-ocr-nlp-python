FROM python:3.9

COPY ["requirements.txt", "/usr/src/app/requirements.txt"]

WORKDIR /usr/src/app

RUN apt update

RUN pip install -r requirements.txt

RUN python -m spacy download es_core_news_sm 

COPY [".", "/usr/src/app"]

CMD tail -f /dev/null