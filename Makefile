pypi: dist
	twine upload dist/*

dist:
	-rm dist/*
	pip install build
	python -m build --sdist

test:
	python -m unittest discover --start-directory tests --pattern "*_test*.py"

clean:
	rm -rf *.egg-info build dist tmp var tests/__pycache__

.PHONY: dist