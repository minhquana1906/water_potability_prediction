# choose the base image for the container
FROM python:3.9-slim

# set the working directory in the container
WORKDIR /app

# copy all files in working space to the container
COPY . /app/
# install the dependencies file to the working directory
RUN pip install -r requirements.txt

EXPOSE 8000

# command to run on container start
CMD ["uvicorn", "abc_test:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]