# PLED Image Enhancement Model

This project contains code for training and testing an image enhancement model.

## Directory Structure

- `train.py`: Model training script
- `test.py`: Model testing script
- `sum_para.py`: Model parameter and FLOPs statistics
- `model/`: Model architecture and loss function definitions
- `DataProcess/`: Data loading and processing
- `utils/`: Utility functions
- `checkpoints/`: Directory for saving model weights
- `log/`: Training logs

## Requirements

- Python 3.8+
- torch 2.4.1
- torchvision

## Training the Model

1. **Prepare the Dataset**  
   Edit `train.py` to set the `train_dir` and `val_dir` paths to your training and validation datasets.

2. **Start Training**  
   Run the following command:

   ```sh
   python train.py --epochs 50000 --batch-size 50 --learning-rate 1e-4
   ```

   You can adjust the parameters as needed. For details, see the `get_args` function in [train.py](train.py).

3. **Logs and Model Saving**  
   - Logs are saved in the `log/` directory.
   - Model weights are saved in the `checkpoints/` directory.

## Testing the Model

1. **Prepare the Test Set**  
   Edit `test.py` to set the data path to your test dataset.

2. **Load Model Weights**  
   By default, the script loads the best model from `checkpoints/model_bestPSNR.pth` or `model_bestSSIM.pth`.

3. **Modify model.py**  
   In the `Illumination_enhance` class, change the return value of the `forward` method to a list (e.g., `[x3]`, or include other intermediate results as needed).

4. **Run Testing**  

   ```sh
   python test.py
   ```

   The test results will be output to the console or a specified directory.

## Model Parameters and FLOPs Statistics

Run:

```sh
python sum_para.py
```

This will output the model's parameter count and FLOPs information.

---

For detailed configuration or custom training/testing workflows, please refer to the comments and function descriptions in each script.