FROM python:3.9.17-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py /app/
COPY saved_model/model.pth /app/saved_model/
COPY templates/index.html /app/templates/

# Expose port 5003
EXPOSE 5003

# Command to run the Flask application
CMD ["python", "app.py"]
