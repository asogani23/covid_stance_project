# Take-Home-LLM-Assessment-

## Results

The model was first evaluated with a zero-shot baseline. It was then fine-tuned on 80% of the data and evaluated on the full dataset of 5,751 tweets to generate the final predictions. Fine-tuning provided a significant improvement in F1-scores.

### Zero-Shot Performance (Baseline)

| Label              | Precision | Recall | F1-Score |
| ------------------ | :-------: | :----: | :------: |
| in-favor           |   0.62    |  0.82  |   0.71   |
| against            |   0.57    |  0.61  |   0.59   |
| neutral-or-unclear |   0.50    |  0.00  |   0.01   |
| **Weighted Avg** | **0.58** | **0.61** | **0.54** |

### Fine-Tuned Performance (Full Dataset)

| Label              | Precision | Recall | F1-Score |
| ------------------ | :-------: | :----: | :------: |
| in-favor           |   0.71    |  0.81  |   0.76   |
| against            |   0.59    |  0.80  |   0.68   |
| neutral-or-unclear |   0.36    |  0.02  |   0.04   |
| **Weighted Avg** | **0.61** | **0.66** | **0.60** |
