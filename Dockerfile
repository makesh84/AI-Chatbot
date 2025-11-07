from python=3.12-slim-buster
workdir /app
copy . /app
run pip install --no-cache-dir -r requirements.txt
cmd ["python", "app.py"]
