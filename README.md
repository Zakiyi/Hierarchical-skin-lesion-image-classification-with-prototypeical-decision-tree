## Hierarchical Skin Lesion Image Classification with Prototypical Decision Tree

This repository provides an implementation of a **hierarchical skin lesion image classification** model using a **prototypical decision tree**. The model integrates deep learning with decision tree-based classification to improve interpretability and accuracy in skin disease diagnosis.

### **Setup & Data Preparation**
#### **1. Prepare Your Dataset**
Ensure that your dataset is structured as a **CSV file** and placed in the `dataset/` directory. The CSV should include:
- Image file paths
- Corresponding labels

#### **2. Download Model Weights**
The pre-trained model weights can be downloaded from [Model Weights (ResNet34)](https://drive.google.com/file/d/11w6_3kdFReIP0jS6017VwXVa555A1qKn/view?usp=sharing) [Model Weights (ViT-B)]([https://drive.google.com/file/d/11w6_3kdFReIP0jS6017VwXVa555A1qKn/view?usp=sharing](https://drive.google.com/file/d/11w6_3kdFReIP0jS6017VwXVa555A1qKn/view?usp=drive_link])

After downloading, place the model weights in the `model_weight/` directory before running any inference or training scripts.

### **Running Inference**
To run inference using the trained model, simply execute the inference.py


### Training your model
If you want to train the model from scratch or fine-tune it, run:
```bash
python main_train.py --metric-guided --loss SoftTreeSupLoss --batch-size 80 --lr-backbone 3e-5 --hierarchy preset-noprune --xwe 0 --analysis SoftEmbeddedDecisionRules --device "cuda:0"
```

#### Main training arguments:
* `--metric-guided` : Enables metric-guided learning.
* `--loss SoftTreeSupLoss` : Uses SoftTreeSupLoss as the loss function.
* `--batch-size 80` : Sets batch size to 80.
* `--lr-backbone 3e-5` : Learning rate for the backbone.
* `--hierarchy preset-noprune` : Uses a pre-set hierarchy without pruning.
* `--alpha 0.4`  : Factor for icd loss.
* `--beta 0.5`  : Factor for metric guided loss.
* `--xw 1` : Weight for cross entropy term.
* `--xwe 0` : Weight for cross entropy term at end of training.
* `--tsw 1` : Weight assigned to tree supervision losses.
* `--tswe 1` : Weight assigned to tree supervision losses at end of training.
* `--analysis SoftEmbeddedDecisionRules` : Enables decision rule analysis.
* `--device "cuda:0"` : Specifies GPU usage (change to "cpu" if needed).

## to do
1. add instruction to build tree graph
2. add jupyter notebook for inference with hierarchy output
