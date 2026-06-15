FROM python:3.9-slim
WORKDIR /app
COPY . /app

# Install AWS CLI via pip instead of apt
RUN pip install --upgrade pip
RUN pip install awscli
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]