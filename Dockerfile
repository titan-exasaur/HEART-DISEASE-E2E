# base image
FROM python:3.10-slim

# set working directory
WORKDIR /app

# copy requirements
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . .

# set pythonpath
ENV PYTHONPATH=/app

# port
EXPOSE 8888

# command
CMD ["uvicorn", "src.heart_disease.app:app", "--host", "0.0.0.0", "--port", "8888"]