#!/bin/bash

echo "===================================================="
echo "		PATH PLANNING PIPELINE"
echo "===================================================="

# cREATE VIRTUAL ENVIORMENT PASTE
if [ ! -d "venv" ]; then
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Impossible to creat venv"
        exit 1
    fi
fi

#Activate virtual enviorment
source venv/bin/activate

#Install packets and dependencies
echo "Installing requirements.txt..."
python3 -m pip install --upgrade pip -q
pip install -r requirements.txt -q

#Execute pipeline with all arguments $@
echo "Initiate pipeline: "
echo "----------------------------------------------------"
python3 pipeline.py "$@"

# 5. Desativar o ambiente virtual ao terminar
deactivate
