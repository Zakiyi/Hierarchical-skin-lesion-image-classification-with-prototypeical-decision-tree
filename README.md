# Hierarchical-skin-lesion-image-classification-with-prototypeical-decision-tree

prepare your data as csv in the dataset
then run the inference.py

The model weights are available [here][https://drive.google.com/your-file-link](https://drive.google.com/file/d/11w6_3kdFReIP0jS6017VwXVa555A1qKn/view?usp=sharing).
Download and place it in the `model_weight/` directory before running the project.

Training

python main_train.py --metric-guided --loss SoftTreeSupLoss --batch-size 80 --lr-backbone 3e-5 --hierarchy preset-noprune --xwe 0 --analysis SoftEmbeddedDecisionRules --device "cuda:0"

# to do
1. add instruction to build tree graph
2. add jupyter notebook for inference with hierarchy output
