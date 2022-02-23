pypi: dist
	twine upload dist/*

dist:
	rm -rf dist/*
	pip install build
	python -m build --sdist

test:
	python -m unittest discover --start-directory tests --pattern "*_test*.py"

clean:
	rm -rf *.egg-info build dist tmp var tests/__pycache__ annlite/*.so

.PHONY: dist
