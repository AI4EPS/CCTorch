# Running CCTorch on cloud

## Running on Kubernetes

### Install volcano
- For  local kubernetes:
```
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml
```
- For GKE:
```
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/v1.5.1/installer/volcano-development.yaml
```
```
kubectl apply -f queue.yaml
```

### Submit jobs
```
torchx run --scheduler kubernetes --scheduler_args namespace=default,queue=test dist.ddp -j 1x4 --script dist_app.py
```

## Running on GCP batch
```
torchx run --scheduler gcp_batch utils.echo --image alpine:latest --msg hello
```
```
torchx status gcp_batch://torchx/earth-beroza:us-central1:echo-rcsmp2d3nxh6f 
```