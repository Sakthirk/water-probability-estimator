# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Copy the current directory contents into the container at /app
COPY /app /app
COPY /model /model

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r app/requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "-u", "app/controller.py"]
