for a1 in tanh
do
for a2 in sigmoid
do
for a3 in base
do


slicer=csw
exp_name=sngan_cifar10_max/${slicer}/${a1}-${a2}-${a3}

python trainmaxsw.py \
    -gen_bs 256 \
    -dis_bs 256 \
    --dataset cifar10 \
    --img_size 32 \
    --max_iter 50000 \
    --model sngan_cifar10 \
    --latent_dim 128 \
    --gf_dim 256 \
    --df_dim 128 \
    --g_spectral_norm False \
    --d_spectral_norm True \
    --g_lr 0.0002 \
    --d_lr 0.0002 \
    --beta1 0.0 \
    --beta2 0.9 \
    --init_type xavier_uniform \
    --n_critic 5 \
    --val_freq 20 \
    --s_lr 0.001 \
    --s_max_iter 100 \
    --use_D \
    --exp_name $exp_name \
    --slicer $slicer \
    --activation $a1 $a2 $a3 \

done
done
done