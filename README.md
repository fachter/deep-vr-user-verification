# Deep VR User Verification

Repository to my Masterthesis ("Motion-Based User Authentication in VR Using Deep Metric Learning with Probabilistic Embeddings")

Parts of the Code originate from my [supervisor](https://github.com/cschell) created for one of his papers "[Versatile User Identification in Extended Reality
using Pretrained Similarity-Learning](https://arxiv.org/pdf/2302.07517)"


# Thesis Content

In my thesis, users were verifies in virtual reality (VR) using their movements with a Deep Metric Learning (DML) approach 
implemented in PyTorch with PyTorch Lightning and PyTorch Metric Learning. 
For monitoring Weights & Biases was used in combination with hydra.

The thesis was written at Julius-Maximilian-University (in Würzburg).

I used the two datasets [Who is Alyx?](https://github.com/cschell/who-is-alyx) and [BOXRR-23](https://rdi.berkeley.edu/metaverse/boxrr-23/),
as well as a newly generated dataset in which my participants had to write words and perform attack scenarios.
All dataset were processed with the [Machine Learning Toolbox](https://github.com/cschell/Motion-Learning-Toolbox).



# Main code parts:

1. [Verification Heads](src/verification_heads)
   1. [Base Class](src/verification_heads/verification_head_base.py)
   2. [Threshold Verification Head](src/verification_heads/threshold_verification_head.py)
   3. [Similarity Verification Head](src/verification_heads/similarity_verification_head.py)
   4. [Match Probability Verification Head (no sampling)](src/verification_heads/distance_match_probability_verification_head.py)
   5. [Match Probability Verification Head (with Monte-Carlo sampling)](src/verification_heads/sampling_match_probability_verification_head.py)
   6. [Mahalanobis Verification Head](src/verification_heads/mahalanobis_verification_head.py)
2. [Evaluator](src/custom_metrics/verification_evaluator.py)
3. [Metric Functions](src/custom_distances)
   1. [KL Divergence](src/custom_distances/kl_divergence_distance.py)
   2. [Verification Cosine Similarity](src/custom_distances/verification_cosine_similarity.py)
   2. [Verification Lp Similarity](src/custom_distances/lp_similarity.py)
4. [Loss Functions](src/custom_losses)
   1. [Soft Contrastive Loss](src/custom_losses/pair_soft_contrastive_loss.py)
   2. [VIB Loss](src/custom_losses/hib_loss.py)
5. [Lightning Module](src/lightning_modules/verification_embedding_module.py)
6. [Base Model](src/models/simple_transformer_model.py)
