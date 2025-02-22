# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the required files to the container
COPY requirements.txt requirements.txt
COPY pcancer.py pcancer.py
COPY final_prostate_cancer_rf_model.pkl final_prostate_cancer_rf_model.pkl
COPY templates/ templates/

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 5001

# Run the Flask app
CMD ["python", "pcancer.py"]
