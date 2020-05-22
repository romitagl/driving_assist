FROM python:3.7

# install production dependencies
COPY dependencies.txt ./
RUN pip install -r dependencies.txt

# add source code
ADD main.py camera.cfg /app/

WORKDIR /app