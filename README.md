# Hierarchical Skin Lesion Image Classification with Prototypical Decision Tree

This repository provides an implementation of a **hierarchical skin lesion image classification** model using a **prototypical decision tree**. The model integrates deep learning with decision tree-based classification to improve interpretability and accuracy in skin disease diagnosis.

## **Setup & Data Preparation**
### **1. Prepare Your Dataset**
Ensure that your dataset is structured as a **CSV file** and placed in the `dataset/` directory. The CSV should include:
- Image file paths
- Corresponding labels

### **2. Download Model Weights**
The pre-trained model weights can be downloaded from the following link:

[Model Weights](https://drive.google.com/file/d/11w6_3kdFReIP0jS6017VwXVa555A1qKn/view?usp=sharing)

After downloading, place the model weights in the `model_weight/` directory before running any inference or training scripts.

## **Running Inference**
To run inference using the trained model, execute the following command:

```bash
python inference.py


## Training the Model
If you want to train the model from scratch or fine-tune it, run:
python main_train.py --metric-guided --loss SoftTreeSupLoss --batch-size 80 --lr-backbone 3e-5 --hierarchy preset-noprune --xwe 0 --analysis SoftEmbeddedDecisionRules --device "cuda:0"

# to do
1. add instruction to build tree graph
2. add jupyter notebook for inference with hierarchy output
