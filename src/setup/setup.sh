#!/bin/bash

conda activate base
conda remove -n SentenceClassification --all --yes
conda env create -f src/setup/requirements.yaml
conda activate SentenceClassification
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3