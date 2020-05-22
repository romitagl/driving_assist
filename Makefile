SHELL := /bin/bash

DOCKER_IMAGE := driving-assist
DOCKER_IMAGE_VERSION := latest

.PHONY: all
all: build_docker

.PHONY: build_docker
build_docker:
	@echo "Building Docker image..."
	docker build -f ./Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_IMAGE_VERSION) .

.PHONY: lucky_spin
lucky_spin:
	xhost +
	docker run --network host --rm -it -v `pwd`/Images:/shared:Z -e DISPLAY=localhost:0.0 -v /tmp/.X11-unix/:/tmp/.X11-unix/:Z $(DOCKER_IMAGE):$(DOCKER_IMAGE_VERSION) python main.py picture /shared/TestImage.png

.PHONY: clean
clean:
	docker rmi -f $(DOCKER_IMAGE):$(DOCKER_IMAGE_VERSION)