
# Multi-Objective-Guided Discrete Flow Matching for Controllable Biological Sequence Design

Designing biological sequences that satisfy multiple, often conflicting, functional and biophysical criteria remains a central challenge in biomolecule engineering. While discrete flow matching models have recently shown promise for efficient sampling in high-dimensional sequence spaces, existing approaches address only single objectives or require continuous embeddings that can distort discrete distributions. We present Multi-Objective-Guided Discrete Flow Matching (MOG-DFM), a general framework to steer any pretrained discrete-time flow matching generator toward Pareto-efficient trade-offs across multiple scalar objectives. At each sampling step, MOG-DFM computes a hybrid rank-directional score for candidate transitions and applies an adaptive hypercone filter to enforce consistent multi-objective progression. We also trained two unconditional discrete flow matching models, PepDFM for diverse peptide generation and EnhancerDFM for functional enhancer DNA generation, as base generation models for MOG-DFM. We demonstrate MOG-DFM's effectiveness in generating peptide binders optimized across five properties (hemolysis, non-fouling, solubility, half-life, and binding affinity), and in designing DNA sequences with specific enhancer classes and DNA shapes. In total, MOG-DFM proves to be a powerful tool for multi-property-guided biomolecule sequence design.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64cd5b3f0494187a9e8b7c69/v4Rr0mhuclD1LN-bWgg2D.png)

## Usage

### 0. Conda Environment

```
conda create -n mog-dfm python=3.9
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install fair-esm transformers xgboost datasets torchdiffeq
```

To use Deep DNAshape, please create another conda environment called `deepDNAshape` following [the guidance of its repository](https://github.com/JinsenLi/deepDNAshape?tab=readme-ov-file#installation).


### 1. PepDFM and EnhancerDFM training and evaluation
The pretrained weights for PepDFM and EnhancerDFM are available in the `ckpt` directory.

The data for PepDFM and EnhancerDFM training are available in the `dataset` directory.

We also provide the complete training and evaluation code for both models.

### 2. Multi-Objective Guided Generation

#### 2.0 Score Models

The pretrained weights for the score models (hemolysis, non-fouling, solubility, half-life, binding affinity, and enhancer class) are available in the `classifier_ckpt` directory. 

Prediction scripts for each score model are provided in the `classifier_code` directory.

#### 2.1 Peptide Generation Task

Example command for peptide generation guided by multiple objectives (hemolysis, non-fouling, solubility, half-life, and binding affinity):
```
python PepDFM_multi_objective_generation.py --is_peptide True --T 100 --n_samples 5 --n_batches 10 --length 10 --target_protein GSHMIEPNVISVRLFKRKVGGLGFLVKERVSKPPVIISDLIRGGAAEQSGLIQAGDIILAVNDRPLVDLSYDSALEVLRGIASETHVVLILRGPEGFTTHLETTFTGDGTPKTIRVTQPLGPPTKAV
```

Note that the hemolysis model outputs one minus the actual hemolysis score, and the half-life model outputs the base-10 logarithm of the half-life in hours.

The guidance settings and their importance weights can be found and modified in `PepDFM_multi_objective_generation.py`

#### 2.2 Enhancer DNA Generation Task

Example command for enhancer DNA generation guided by the enhancer class and DNA shape:
```
python EnhancerDFM_multi_objective_generation.py --is_peptide False --T 800 --n_samples 5 --n_batches 10 --length 100 --target_enhancer_class 0 --target_DNA_shape HelT
```

The guidance settings and their importance weights can be found and modified in `EnhancerDFM_multi_objective_generation.py`

To use this repository, you agree to abide by the [MOG-DFM License](https://drive.google.com/file/d/1LJuGrsRZMoqsrZa1gSfsCihiih5MPVRA/view?usp=sharing).
