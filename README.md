# TremorShield 🖱️

### An Adaptive Context-Aware Cursor Stabilization Framework

TremorShield is an assistive technology project designed to improve computer accessibility for individuals experiencing hand tremors and motor impairments.

The system aims to provide smoother, more controlled and stable cursor movement by combining machine learning-based interaction recognition with an adaptive Kalman filtering approach.

Unlike conventional cursor stabilization techniques that apply the same filtering behaviour continuously, TremorShield aims to adapt cursor stabilization according to the user's movement patterns and interaction context.

---

## Project Overview

People experiencing hand tremors may find everyday computer interactions challenging, particularly when performing tasks that require precise cursor movements, such as clicking small targets, selecting text, dragging objects and navigating interfaces.

TremorShield addresses this challenge through a modular framework consisting of:

* Mouse interaction data collection.
* Data cleaning and preprocessing.
* Synthetic tremor generation.
* Data validation and analysis.
* Machine learning-based interaction classification.
* Adaptive Kalman filter integration.
* A downloadable desktop application.
* A public website for accessing and downloading the application.

The ultimate objective is to develop an accessible desktop solution that can adapt cursor behaviour to different interaction requirements.

---

## System Architecture

The project is organized into eight major development phases.

| Phase | Module                      | Description                                                                                     |
| ----- | --------------------------- | ----------------------------------------------------------------------------------------------- |
| 01    | Data Collection             | Collects mouse movement and interaction data.                                                   |
| 02    | Data Cleaning               | Processes collected data and prepares it for subsequent stages.                                 |
| 03    | Synthetic Tremor Generation | Generates synthetic tremor patterns for experimentation and model development.                  |
| 04    | Data Validation             | Evaluates generated data using signal, spatial and statistical validation techniques.           |
| 05    | ML Models                   | Develops and evaluates machine learning models for interaction classification.                  |
| 06    | Adaptive Kalman Filter      | Integrates adaptive filtering for context-aware cursor stabilization.                           |
| 07    | Desktop Application         | Provides the integrated cursor stabilization system through an installable desktop application. |
| 08    | Website                     | Provides project information and a platform for downloading the desktop application.            |

### Overall Workflow

```text
       Mouse Interaction Data
                 |
                 v
          Data Collection
                 |
                 v
           Data Cleaning
                 |
                 v
     Synthetic Tremor Generation
                 |
                 v
          Data Validation
                 |
                 v
       Machine Learning Models
                 |
                 v
     Adaptive Kalman Integration
                 |
                 v
       Desktop Application
                 |
                 v
          User Interface
                 ^
                 |
          Download Website
```

---

## Repository Structure

```text
TremorShield/
│
├── 01_DATA_COLLECTION/
│
├── 02_DATA_CLEANING/
│
├── 03_SYNTHETIC_TREMOR_GENERATION/
│
├── 04_DATA_VALIDATION/
│
├── 05_ML_MODELS/
│
├── 06_ADAPTIVE_KALMAN_FILTER/
│
├── 07_DESKTOP_APPLICATION/
│
├── 08_WEBSITE/
│
├── 99_ARCHIVE_REVIEW/
│
├── docs/
│
├── PROJECT_STRUCTURE.md
│
└── README.md
```

Each module is maintained separately to support modular development, testing, integration and future improvements.

---

## Development Phases

### 01. Data Collection

This phase focuses on collecting mouse interaction data through a web-based data collection interface.

The collection system records mouse-related information, including:

* Cursor coordinates.
* Movement direction.
* Displacement.
* Velocity.
* Acceleration.
* Button interactions.
* Clicking and dragging events.
* Task and session information.
* Target-related information.

The collected information provides the foundation for subsequent preprocessing and model development.

**Technology:** Python, FastAPI, WebSockets, HTML, CSS and JavaScript.

### 02. Data Cleaning

The data cleaning phase prepares the collected mouse interaction data for downstream processing.

Its intended responsibilities include:

* Handling missing or invalid values.
* Removing duplicate records where appropriate.
* Checking data consistency.
* Preparing movement sequences.
* Ensuring compatibility with subsequent processing stages.

### 03. Synthetic Tremor Generation

This phase focuses on generating synthetic tremor patterns and incorporating them into cursor movement data.

The synthetic generation pipeline supports experimentation with different tremor characteristics and movement conditions.

Its responsibilities include:

* Loading and preparing input data.
* Participant anonymization.
* Data splitting.
* Synthetic tremor modelling.
* Tremor injection into movement trajectories.
* Feature generation.
* Exporting processed datasets.

### 04. Data Validation

This phase evaluates the generated data to determine whether the synthetic tremor patterns and movement characteristics are consistent with the intended modelling requirements.

Validation includes:

* Signal-level validation.
* Spatial movement validation.
* Statistical comparison.
* Reference-based movement analysis.
* Validation reporting.

RobustPDX reference data is used for movement and spatial validation where available. This does not make Parkinson's disease diagnosis or prediction a function of TremorShield.

### 05. Machine Learning Models

This phase develops machine learning models to identify the user's current mouse interaction context.

The classification task focuses on six interaction categories:

| Class            | Description                                 |
| ---------------- | ------------------------------------------- |
| Navigation       | General cursor navigation.                  |
| Clicking         | Single and double-click interactions.       |
| Dragging         | Drag-related cursor movements.              |
| Precision        | Precision-oriented interactions.            |
| Target Selection | Movement associated with selecting targets. |
| Idle             | Periods of inactivity.                      |

The current model comparison includes:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest

Model evaluation includes classification metrics, prediction outputs, confusion matrices and feature importance analysis where applicable.

The purpose of this phase is to provide interaction context that can support adaptive cursor stabilization.

### 06. Adaptive Kalman Filter Integration

This phase focuses on integrating an adaptive Kalman filtering mechanism into the TremorShield framework.

The objective is to reduce unwanted cursor fluctuations while preserving intentional user movement.

The intended integration will use movement information and interaction context to adapt cursor stabilization behaviour.

Key development objectives include:

* Real-time cursor movement processing.
* Tremor-related movement estimation.
* Adaptive filtering.
* Context-aware parameter adjustment.
* Balancing cursor smoothness and responsiveness.
* Integration with machine learning model outputs.

### 07. Desktop Application

The desktop application will serve as the primary user-facing implementation of TremorShield.

It is intended to combine the machine learning models, adaptive Kalman filtering and cursor processing into an installable software application.

Planned capabilities include:

* User-friendly interface.
* Cursor stabilization controls.
* Configuration and customization.
* Machine learning integration.
* Adaptive Kalman filter integration.
* Real-time cursor stabilization.
* Application packaging and installation.

The application will be distributed through the TremorShield website.

### 08. Website

The website will act as the public-facing platform for TremorShield.

Its intended functionality includes:

* Project introduction.
* Explanation of system features.
* Documentation and usage information.
* Application download page.
* Desktop application installation instructions.
* Project updates and support information.

The website and desktop application are separate components: the website distributes the software, while the installed desktop application performs the actual cursor stabilization.

---

## Technology Stack

The project currently uses or is intended to use the following technologies.

| Category                | Technologies          |
| ----------------------- | --------------------- |
| Programming Language    | Python                |
| Machine Learning        | Scikit-learn          |
| Data Processing         | Pandas, NumPy         |
| Scientific Computing    | SciPy                 |
| Backend                 | FastAPI               |
| Real-Time Communication | WebSockets            |
| Frontend                | HTML, CSS, JavaScript |
| Data Visualization      | Matplotlib            |
| Version Control         | Git and GitHub        |

Additional technologies may be introduced during adaptive filtering, desktop application development and website development.

---

## Project Status

TremorShield is being developed in modular phases.

| Phase                       | Current Status                   |
| --------------------------- | -------------------------------- |
| Data Collection             | Implemented                      |
| Data Cleaning               | Dedicated implementation pending |
| Synthetic Tremor Generation | Implemented                      |
| Data Validation             | Implemented                      |
| Machine Learning Models     | Initial models implemented       |
| Adaptive Kalman Filter      | Integration phase                |
| Desktop Application         | Planned integration              |
| Website                     | Planned development              |

The project is under active development. Features described as planned or in progress should not be considered completed functionality.

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/ryukkz/TremorShield.git
```

```bash
cd TremorShield
```

### Set Up the Machine Learning Environment

Navigate to the ML models module:

```bash
cd 05_ML_MODELS
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Individual modules may have their own dependencies and execution instructions. Refer to their respective documentation before running the pipelines.

---

## Repository Guidelines

To maintain a clean and manageable project:

* Keep each development phase within its designated directory.
* Separate source code, datasets, model outputs and documentation.
* Maintain individual directories for each machine learning model.
* Store model-specific metrics and evaluation results alongside the corresponding model.
* Avoid committing temporary files, unnecessary logs and generated cache files.
* Keep large datasets and model artifacts organized separately from source code.
* Document changes when integrating different modules.

---

## Future Scope

Future development will focus on completing the end-to-end TremorShield system.

Potential improvements include:

* More advanced tremor estimation.
* Improved adaptive filtering techniques.
* Real-time context-aware cursor stabilization.
* Personalized stabilization settings.
* Cross-platform desktop support.
* Installer and executable generation.
* Website-based application distribution.
* Usability testing and performance evaluation.

---

## Project Repository

**GitHub:** https://github.com/ryukkz/TremorShield

---

## Vision

TremorShield aims to make everyday computer interaction more accessible by developing an intelligent cursor stabilization system that adapts to the user's movement and interaction requirements.

The long-term vision is to provide a practical, customizable and user-friendly assistive technology solution that helps individuals with motor impairments interact with computers more comfortably and confidently.
