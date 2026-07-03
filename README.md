# Rentgen Detector

Jupyter Notebook project for chest X-ray classification: `NORMAL` and `PNEUMONIA`.

## Stack

- Python
- Keras / TensorFlow
- OpenCV
- NumPy
- scikit-learn

## Project Structure

```text
rentgen-detector/
|-- rentgen_detector.ipynb
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

## Run

```bash
jupyter notebook rentgen_detector.ipynb
```

The notebook expects the dataset folder near the notebook:

```text
images/train
images/test
```

## What Is Inside

- class weights for imbalanced data
- early stopping by validation loss
- grayscale image preprocessing
- simple CNN architecture
- clean Jupyter notebook without saved outputs

## Classes

```python
NORMAL = 0
PNEUMONIA = 1
```
