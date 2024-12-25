## Quick Run


## Pre-training AI Tasks - Image Classification
```
python GoogleNet_train.py
```
## train
The payload can be changed from 0.1bpp to 1.0bpp:

The parameters of the trained GoogleNet model are the '_google_net_add.pkl_'
```
python train.py --bpp 1.0 --kernel_size1 3 --kernel_size2 3 --num_epochs 150
```
If breakpoint training is required:
```
python train.py --bpp 1.0 --kernel_size1 3 --kernel_size2 3 --num_epochs 150 --continue_train 1 --continue_epoch 0
```

## val
The parameters of the trained model are in the '_model_' folder
```
python val.py --bpp 1 --kernel_size1 3 --kernel_size2 3 --best_epoch 149
```

## test
Generate Picture:
```
python test_pic.py --bpp 1.0 --kernel_size1 3 --kernel_size2 3 --best_epoch 149
```
Resistance to Steganalysis SrNet:
```
python test_SrNet_train_val.py  --bpp 1.0 --kernel_size1 3 --kernel_size2 3 --num_epochs 150 --best_epoch 149
```
```
python test_SrNet_ROC.py  --bpp 1.0 --kernel_size1 3 --kernel_size2 3 --num_epochs 150 --best_epoch 149
```
Resistance to Steganalysis XuNet:
```
python test_XuNet_train_val.py  --bpp 1.0 --kernel_size1 3 --kernel_size2 3 --num_epochs 150 --best_epoch 149
```
```
python test_XuNet_ROC.py  --bpp 1.0 --kernel_size1 3 --kernel_size2 3 --num_epochs 150 --best_epoch 149
```