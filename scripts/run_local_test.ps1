# Run from the repository root on Windows PowerShell
# First time only:
#   python -m venv .venv
#   .venv\Scripts\activate
#   python -m pip install --upgrade pip
#   pip install -r requirements.txt

.venv\Scripts\activate
python -m src.train --dataset GunPoint --representation gaf --model_type light2dcnn --epochs 2 --batch_size 16 --image_size 64 --seed 42
