
IMAGE_REPO ?= ghcr.io/harakas/models/yolov8
TAG ?= latest

all: build

build:
	docker build -t $(IMAGE_REPO):$(TAG) .
	docker save $(IMAGE_REPO):$(TAG)|tar --to-stdout -xf - --wildcards '*/layer.tar'|gzip -9 - > yolov8.tar.gz

push: build
	docker push $(IMAGE_REPO):$(TAG)

