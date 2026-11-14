# lightweight Python image
FROM python:3.11-slim

# working dir inside container
WORKDIR /app

# copy dependencies file to working dir
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy project files
COPY . .

# make port 8000 available
EXPOSE 8000

# start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]