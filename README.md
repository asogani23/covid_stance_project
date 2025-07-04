# Take-Home-LLM-Assessment-

## Results

The model was first evaluated with a zero-shot baseline. The initial fine-tuning showed a weakness in identifying the "neutral-or-unclear" class due to class imbalance. By implementing an oversampling strategy for the training data, the model's performance on this minority class improved dramatically.

### Zero-Shot Performance (Baseline)

| Label              | Precision | Recall | F1-Score |
| ------------------ | :-------: | :----: | :------: |
| in-favor           |   0.62    |  0.82  |   0.71   |
| against            |   0.57    |  0.61  |   0.59   |
| neutral-or-unclear |   0.50    |  0.00  |   0.01   |
| **Weighted Avg** | **0.58** | **0.61** | **0.54** |


### Initial Fine-Tuned Performance

| Label              | Precision | Recall | F1-Score |
| ------------------ | :-------: | :----: | :------: |
| in-favor           |   0.71    |  0.81  |   0.76   |
| against            |   0.59    |  0.80  |   0.68   |
| **neutral-or-unclear** | **0.36** |  **0.02** |   **0.04** |
| **Weighted Avg** | **0.61** | **0.66** | **0.60** |


### Final Performance (with Oversampling)

| Label              | Precision | Recall | F1-Score |
| ------------------ | :-------: | :----: | :------: |
| in-favor           |   0.80    |  0.76  |   0.78   |
| against            |   0.58    |  0.84  |   0.69   |
| **neutral-or-unclear** | **0.58** |  **0.26** |   **0.36** |
| **Weighted Avg** | **0.69** | **0.69** | **0.67** |
