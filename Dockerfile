# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port the app runs on
EXPOSE $PORT

# Define environment variable
ENV FLASK_APP=pcancer.py
ENV PYTHONUNBUFFERED=1  # Prevents output buffering

# Run the Flask app with Gunicorn (Heroku recommended)
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "pcancer:app"]

