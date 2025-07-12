rm -rf venv
/usr/local/bin/python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python fuel_efficiency_predictor.py
