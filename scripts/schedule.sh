#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

#python3 src/train.py experiment=verification_head_search/match_probability/euclidean_distance_soft_contrastive_loss.yaml
#
#python3 src/train.py experiment=verification_head_search/match_probability/manhattan_distance_soft_contrastive_loss.yaml
#
#python3 src/train.py experiment=verification_head_search/match_probability/cosine_similarity_soft_contrastive_loss.yaml

python3 src/train.py experiment=verification_head_search/match_probability/kl_div_soft_contrastive_loss.yaml
