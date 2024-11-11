# First install the python 3.13, the slim version uses less space
FROM python:3.13.0-bullseye

# Install pipenv library in Docker 
RUN pip install gunicorn pipenv markupsafe numpy scikit-learn==1.5.2 werkzeug blinker

WORKDIR /. 

# Copy the Pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker 
COPY ["*.py", "*.bin","./"]

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

# If we run the Docker image, we want our  app to be running
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]