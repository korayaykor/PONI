#!/bin/bash

EXPT_ROOT=$PWD

#conda activate poni

cd $PONI_ROOT
#CUDA_VISIBLE_DEVICES=1
python -W ignore -u -m torch.distributed.launch \
  --use_env \
  --nproc_per_node=1 \
  train.py \
    LOGGING.ckpt_dir "$EXPT_ROOT/checkpoints" \
    LOGGING.tb_dir "$EXPT_ROOT/tb" \
    LOGGING.log_dir "$EXPT_ROOT" \
    DATASET.root $PONI_ROOT/data/semantic_maps/mp3d/precomputed_dataset_24.0_123_spath_square \
    DATASET.dset_name 'mp3d' \
    DATASET.enable_unexp_area True \
    DATASET.object_pf_cutoff_dist 5.0 \
    DATASET.normalize_area_by_constant True \
    DATASET.max_unexp_area 40.0 \
    OPTIM.batch_size 32 \
    MODEL.num_categories 23
