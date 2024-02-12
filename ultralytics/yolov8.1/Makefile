
IMAGE_REPO ?= ghcr.io/harakas/models/yolov8.1
TAG ?= 1.2

all: build

build:
	docker build -t $(IMAGE_REPO):$(TAG) .

push: build
	docker push $(IMAGE_REPO):$(TAG)

dist: build
	rm -rf dist && mkdir -p dist && cd dist && docker save $(IMAGE_REPO):$(TAG)|tar --to-stdout -xf - --wildcards '*/layer.tar'|tar xvf -
	cd dist && rm -f yolov8.small.models.tar.gz && tar zcf yolov8.small.models.tar.gz LICENSE yolov8[ns]*320x320* labels*

release: build dist
	gh release create yolov8.1-$(TAG) dist/*

