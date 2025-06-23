# Load2UnNet

Graph neural network for predicting unloaded cardiac geometry from end-diastolic meshes.

## Method

- **Architecture**: Graph Attention Network with cycle consistency
- **Input**: 3D cardiac mesh + simulation parameters 
- **Output**: Unloaded cardiac geometry
- **Loss**: Forward deformation + cycle consistency

## Installation

```bash
pip install torch torch-geometric vtk numpy scipy matplotlib pyvista
```

## Dataset Structure

```
dataset/
├── train/
│   ├── ED/           # End-diastolic meshes
│   └── unloaded/     # Unloaded reference meshes
├── val/
│   ├── ED/
│   └── unloaded/
└── test/
    ├── ED/
    └── unloaded/
```

Files: `{type}{id}_{pressure}_{stiffness}_{endo_helix}_{epi_helix}.vtu`

## Usage

```bash
# Training
python model_A0.py

# Evaluation only
python model_A0.py --eval
```

## Configuration

Adjust weak supervision in `model_A0.py`:
```python
TRAINING_CONFIG = {
    'WEAK_SUPERVISION_RATIO': 0.2  # 20% weak supervision
}
```

## Output

- Predicted meshes: `model_A0/predicted_unloaded/`
- Performance metrics: DICE, Hausdorff distance, mean distance

## Citation

*Manuscript is currently under preparation and will be submitted for publication in the near future*

```
