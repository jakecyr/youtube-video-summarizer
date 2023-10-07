.PHONY=build run

build:
	docker build -t python-poetry .

run: build
run:
	docker run --rm -it python-poetry /bin/sh
