# Use the official Python image
FROM python:3.11.4

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file first to leverage Docker cache
COPY requirements.txt .

# Install required Python packages
RUN pip install -r requirements.txt --default-timeout=100 future

# Copy the rest of the application files to the container's working directory
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8000

# Command to run your Django application
CMD python manage.py runserver