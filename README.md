# TREMORSHIELD
### An Adaptive Context-Aware Cursor Stabilization Framework for Motor-Impaired Computer Users

TREMORSHIELD aims to improve mouse interaction for people with hand tremors by exploring machine learning-based mouse movement classification and adaptive cursor stabilization.

## Project Modules

### 1. `DATA_COLLECTION`
A data collection website used to record mouse coordinates and movement data while users perform different interaction tasks.

### 2. `MLmodels`
Contains the machine-learning pipeline for processing mouse movement data, training multiple classification models, and comparing their performance across six interaction classes:

- Navigation
- Clicking
- Dragging
- Selection
- Precision
- Idle

### 3. `TremorShield-Application`
Contains a small HTML-based frontend demo demonstrating the project's application interface.

## Project Workflow

1. Collect mouse movement data through the data collection website.
2. Preprocess the recorded data and extract movement features.
3. Train and compare machine-learning classification models.
4. Explore the integration of interaction classification with adaptive cursor stabilization.

## Project Status

The data collection website, machine-learning model comparison, and basic frontend demo are included in the repository. Integration of the trained model with real-time cursor stabilization and further system evaluation remain future development tasks.

## Technologies

- Python and machine-learning libraries for data processing and model comparison.
- HTML and web technologies for the data collection website and frontend demo.
- CSV files for storing mouse movement data and extracted features.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/CereenaG/TremorShield.git
cd TremorShield
```

Explore the `DATA_COLLECTION`, `MLmodels`, and `TremorShield-Application` directories for their respective components.

Additional setup instructions depend on the requirements of each module.

