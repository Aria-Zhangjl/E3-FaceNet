export OUTDIR='OUTPUT_DIR'
export CHECKPOINT_PATH='/home/xmu/zjl/code/E3-FaceNet/ckpt/_final_model.pkl'

export SEEDS='42'
CUDA_VISIEBLE_DEVICES=6 python sample.py \
    --outdir=$OUTDIR --trunc=0.7 --seeds=$SEEDS --network=$CHECKPOINT_PATH \
    --render-program="rotation_camera" \
    --gen_description='The woman has wavy hair.' \
    --edit_description='This person has beard.' \
    --alpha='0.5' \
    --no-video=True \
    --relative_range_u_scale=2.0 \
    --n_steps=7 \