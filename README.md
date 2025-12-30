# ML-Based Power Prediction from Synthesized Netlists

This repository contains an end-to-end machine learning pipeline for early-stage power estimation using synthesized gate-level netlists.

The objective of this project is to enable fast, approximate power estimation before signoff analysis by learning correlations between netlist structure and tool-reported power metrics.

---

## Project Overview

The workflow followed in this project is:

Verilog RTL  
→ Synthesis using Cadence Genus  
→ Gate-level netlist + power reports  
→ Feature extraction from netlists  
→ Machine learning–based power prediction  

Power metrics are extracted directly from Cadence Genus after synthesis and used as ground truth labels for training.

---

## Files in This Repository

- `genus_power_dataset.xlsx`  
  Contains power metrics extracted from Cadence Genus, including:
  - Leakage power  
  - Internal power  
  - Switching power  
  - Total power  
  - Cell count  
  - Total area  

  Each row corresponds to a synthesized Verilog design.

- `netlist_power_prediction.py`  
  Implements the complete machine learning pipeline:
  - Parsing of synthesized gate-level netlists
  - Structural and instance-based feature extraction
  - Hierarchical modeling of area, cell count, and power
  - Training and evaluation of regression models
  - Power prediction for unseen netlists

---

## Methodology

### 1. Synthesis and Data Generation
Verilog RTL designs are synthesized using **Cadence Genus**. Power metrics are obtained using synthesis-time default activity factors.

### 2. Feature Extraction
Features are derived directly from synthesized netlists, including:
- Standard-cell instance counts
- Structural statistics (token count, unique tokens, netlist length)
- Sparse token-based representations of netlist content

These features capture structural and connectivity-related information relevant to power consumption.

### 3. Machine Learning Models
The following models are used:
- **Linear Regression (positive constraint)** for area estimation from instance counts
- **Random Forest Regression** for cell count prediction
- **Multi-output Random Forest Regression** for predicting:
  - Leakage power
  - Internal power
  - Switching power
  - Total power

Power targets are log-transformed during training to handle skewed distributions.

---

## Assumptions and Limitations

- Uses synthesis-time default activity factors (no VCD or SAIF-based switching activity)
- Intended for **relative and early-stage power estimation**, not signoff-level accuracy
- Model accuracy depends on the diversity of designs and synthesis constraints in the training dataset

---

## How to Run

Install required Python packages:

```bash
pip install numpy pandas scikit-learn scipy joblib openpyxl
