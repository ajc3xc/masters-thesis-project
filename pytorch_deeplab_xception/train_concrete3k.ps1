#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
$env:PATH = "D:\camerer_ml\cudnn\cudnn-windows-x86_64-8.9.7.29_cuda11-archive\bin;" + $env:PATH
$env:PATH = "D:\camerer_ml\.pixi\envs\default\Lib\site-packages\nvidia\cublas\bin;" + $env:PATH
pixi run python train.py --backbone xception --dataset concrete3k --checkname concrete3k_xception_ft --epochs 20 --base-size 500 --crop-size 500 --batch-size 8 --lr 0.0001 --gpu-ids 0 --resume C:\\Users\\ajc3xc\\ml\\pytorch_deeplab_xception\\run\\crackseg9k\\crackseg9k_no_pretrain\\experiment_3\\checkpoint.pth.tar --ft