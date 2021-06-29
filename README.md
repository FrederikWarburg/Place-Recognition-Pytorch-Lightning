# Place-Recognition-Pytorch


## Train on MSLS

```
CUDA_VISIBLE_DEVICES=0 python train.py --directory '.' --data_dir /scratch/frwa/MSLS/ --train_cities "zurich" --workers 8 --val_cities "" --test_cities "" --batch_size 16 --arch resnet50
```

## Validation on MSLS