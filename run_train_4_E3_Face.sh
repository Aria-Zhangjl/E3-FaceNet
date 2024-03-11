export OUTDIR='OUTPUT_DIR'
export DATADIR='DATADIR'
export SPEC='paper512_mmceleba'
export MODEL='model_4_E3_Face'
export RESUME='STYLENERF_CHECKPOINT'
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_train_4_E3_Face.py outdir=$OUTDIR data=$DATADIR spec=$SPEC model=$MODEL resolution=512 resume=$RESUME resume_run=True smooth_feat_v3=True