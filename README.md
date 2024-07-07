# KG-EnPred

A PyTorch implementation using a knowledge graph to predict when and which vehicles will enter a highway toll station.

## VERSION: 0.0.0

the best metric

|       | MRR    | hits@1 | hits@3 | hits@10 |
| ----- | ------ | ------ | ------ | ------- |
| TRAIN | 67.68% | 55.66% | 76.76% | 87.38%  |
| VALID | 62.04% | 50.13% | 70.82% | 81.50%  |
| TEST  | 63.60% | 51.73% | 72.48% | 82.95%  |

## VERSION: 0.0.1

the best metric

update the strategy of updating entity embedding.

loss decreases faster and has smaller fluctuations.

|       | MRR    | hits@1 | hits@3 | hits@10 |
| ----- | ------ | ------ | ------ | ------- |
| TRAIN | 79.56% | 69.02% | 88.92% | 96.22%  |
| VALID | 67.03% | 55.79% | 76.05% | 86.28%  |
| TEST  | 69.22% | 58.23% | 78.08% | 87.90%  |
