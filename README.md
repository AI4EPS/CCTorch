Cross-correlation using Pytorch

## Single GPU
python cc.py

## Multi GPU
torchrun --standalone --nproc_per_node=8 cc.py