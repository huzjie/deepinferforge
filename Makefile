PY ?= python

.PHONY: all doctor bench train test clean serve

all: test doctor bench

doctor:
	$(PY) -m deepinferforge doctor

bench:
	$(PY) -m deepinferforge bench

train:
	$(PY) -m deepinferforge train

compile:
	$(PY) -m deepinferforge compile

serve:
	$(PY) -m deepinferforge serve

test:
	$(PY) -m unittest discover -s tests -t .

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
