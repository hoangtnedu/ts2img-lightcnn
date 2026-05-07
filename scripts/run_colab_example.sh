# Run from repository root on Google Colab after mounting Drive and installing requirements.
python -m src.train \
  --dataset GunPoint \
  --representation gaf \
  --model_type light2dcnn \
  --epochs 50 \
  --batch_size 32 \
  --image_size 64 \
  --seed 42
