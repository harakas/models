
IMAGE_REPO ?= ghcr.io/harakas/models/yolov8.1
TAG ?= 1.0

all: build

build:
	docker build -t $(IMAGE_REPO):$(TAG) .

push: build
	docker push $(IMAGE_REPO):$(TAG)

release: build
	rm -rf dist && mkdir -p dist && cd dist && docker save $(IMAGE_REPO):$(TAG)|tar --to-stdout -xf - --wildcards '*/layer.tar'|tar xvf -
	gh release create yolov8.1-$(TAG) dist/*

