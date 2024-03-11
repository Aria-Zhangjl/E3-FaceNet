export OUTDIR='OUTPUT_DIR'
export DATADIR='DATA_SET'
export SPEC='paper512_mmceleba'
export MODEL='model_E3_face'
export RESUME='MODEL CHECKPOINT'
CUDA_VISIBLE_DEVICES=0 python run_eval_4_E3_Face.py outdir=$OUTDIR data=$DATADIR spec=$SPEC model=$MODEL resolution=512 resume=$RESUME resume_run=True