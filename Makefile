.DEFAULT_GOAL := default

help:
	@echo "clean - remove Python file artifacts"
	@echo "init - initialize app (config and Python dependencies)"

all: default

default: clean init

clean:
	find ./ -name '*.pyc' -exec rm -f {} +
	find ./ -name '*.pyo' -exec rm -f {} +
	find ./ -name '__pycache__' -exec rm -fr {} +

init:
	echo "Copying configuration files"
	cp -n config.example.json config.json || true
	cp -n logging.example.json logging.json || true
	echo "Installing dependencies"
	pip install --upgrade pip && pip install -r requirements.txt
	echo "Downloading NTLK and SpaCy data"
	python -m nltk.downloader stopwords
	python -m spacy download en_core_web_sm
	echo "Downloading embeddings"
	cd models/embeddings/ && wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip
	cd models/embeddings/ && rm glove.6B.zip
	echo "Done"