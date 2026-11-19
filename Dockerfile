# lightweight Python image (slim- smaller version of image)
FROM python:3.11-slim

# working dir inside container for subsequent commands
WORKDIR /app

# copy dependencies file to working dir (/app)
COPY requirements.txt .

# install dependencies inside image
RUN pip install --no-cache-dir -r requirements.txt

# copy project files on computer into image
COPY . .

# make computer's port 8000 available
EXPOSE 8000

# start FastAPI app
# what command to run inside container when it starts
# app:app refers to app FastAPI object inside app.py file
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]