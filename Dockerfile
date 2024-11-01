
FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/test_multitask_model.py"]
