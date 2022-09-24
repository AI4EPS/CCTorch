# CCTorch: Cross-Correlation using Pytorch

## Single GPU
python run.py

## Multi GPU (e.g., using 8 GPUs)
torchrun --standalone --nproc_per_node=8 run.py
