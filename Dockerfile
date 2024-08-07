# Name the single Python image we're using everywhere.
ARG python=python:3.10-slim-bullseye

# Build stage:
FROM ${python} AS build

# Install a full C toolchain and C build-time dependencies for
# everything we're going to need.
RUN apt-get update \
   && DEBIAN_FRONTEND=noninteractive \
   apt-get install --no-install-recommends --assume-yes \
   build-essential \
   libpq-dev

# Create the virtual environment.
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install the Python library dependencies, including those with
# C extensions.  They'll get installed into the virtual environment.
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Final stage:
FROM ${python} 

WORKDIR /app

# Copy the virtual environment from the first stage.
COPY --from=build /venv /venv
ENV PATH=/venv/bin:$PATH

RUN apt-get update \
   && DEBIAN_FRONTEND=noninteractive \
   apt-get install --no-install-recommends --assume-yes \
   libgomp1 \
   && rm -rf /var/lib/apt/lists/* 

# Copy the application in.
COPY . .
# Specify whether the port listens
EXPOSE 5000

ENTRYPOINT ["python", "handler.py"]