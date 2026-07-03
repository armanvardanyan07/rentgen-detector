# Rentgen Detector

CNN model for chest X-ray classification: `NORMAL` and `PNEUMONIA`.

## Stack

- Python
- Keras / TensorFlow
- OpenCV
- NumPy
- scikit-learn

## Project Structure

```text
rentgen-detector/
|-- train.py
|-- predict.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

Dataset is not stored in this repository. Put images in this format:

```text
images/
|-- train/
|   |-- NORMAL/
|   `-- PNEUMONIA/
`-- test/
    |-- NORMAL/
    `-- PNEUMONIA/
```

## Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --data ../images
```

After training, the model will be saved as:

```text
xray_model.keras
```

## Predict

```bash
python predict.py --image ../images.jpg
```

## What Is Inside

- class weights for imbalanced data
- early stopping by validation loss
- grayscale image preprocessing
- simple CNN architecture
- clean train and predict scripts

## Classes

```python
NORMAL = 0
PNEUMONIA = 1
```
