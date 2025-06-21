# Use a standard Python 3.9 image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy all the project files into the container
COPY . .

# Install the project and its dependencies from pyproject.toml
# This command assumes your pyproject.toml is set up for a standard install.
RUN pip install --no-cache-dir .

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application using uvicorn
# The host must be 0.0.0.0 to be accessible from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 