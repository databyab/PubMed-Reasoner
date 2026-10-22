.PHONY: install test ui
install:
	python -m pip install -e '.[dev]'
test:
	pytest
ui:
	streamlit run app.py

