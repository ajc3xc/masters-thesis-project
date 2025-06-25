CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --backbone resnet --lr 0.01 --workers 4 --epochs 40 --batch-size 16 --gpu-ids 0,1,2,3 --checkname deeplab-resnet --eval-interval 1 --dataset coco
sbatch --account=general --partition=gpu --time=1:00:00 --gres=gpu:1 --cpus-per-task=1 --mem=4G --output=train_%j.out --wrap="echo 'Running original gbc'; exit"
