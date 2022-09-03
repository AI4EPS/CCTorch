Cross-correlation using Pytorch

## Single GPU
python run.py

## Multi GPU
torchrun --standalone --nproc_per_node=8 run.py
