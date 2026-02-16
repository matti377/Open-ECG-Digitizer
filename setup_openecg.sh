#!/bin/bash

/opt/homebrew/bin/python3.12 -m venv openecg-env
source openecg-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

