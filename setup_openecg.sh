#!/bin/bash

/opt/homebrew/bin/python3.12 -m venv openecg-env
source openecg-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

#use
#python3 -m src.digitize --config c:\Users\matti\Downloads\Open-ECG-Digitizer\src\config\inference_wrapper.yml      