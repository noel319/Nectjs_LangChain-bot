FROM python:3.11.5

EXPOSE 5328

WORKDIR /flask

COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

ENV FLASK_DEBUG=1

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "-p", "5328"]