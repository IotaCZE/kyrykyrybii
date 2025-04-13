# kyrykyrybii

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Last Commit](https://img.shields.io/github/last-commit/IotaCZE/kyrykyrybii)
![Issues](https://img.shields.io/github/issues/IotaCZE/kyrykyrybii)
![Stars](https://img.shields.io/github/stars/IotaCZE/kyrykyrybii?style=social)
![Repo Size](https://img.shields.io/github/repo-size/IotaCZE/kyrykyrybii)

A data processing and visualization toolkit for medical imaging datasets, with a focus on CT scans and radiotherapy planning data.

## 📁 Project Structure

- **`Rackaton_Data/SAMPLE_00X/`**: Directories containing data for individual subjects, where `00X` denotes the subject ID.
- **`playbook.ipynb`**: Main notebook demonstrating data processing workflows.
- **`ct_series.py`**: Script for handling CT series data.
- **`data_parsing.py`**, **`data_parsing_merged.py`**: Scripts for parsing and merging dataset information.
- **`measure.py`**: Contains functions for measuring and analyzing data metrics.
- **`show_dcm.py`**: Utility for displaying DICOM images.
- **`visualize_mask.py`**: Tool for visualizing segmentation masks.
- **`requirements.txt`**: List of Python dependencies.
- **`target_contours.txt`**: Text file listing target contours for analysis.

## 🧾 Scan Prefixes

- **RT**: Radiotherapy treatment records.
- **RS**: Structure sets.
- **RI**: To be ignored.
- **RE**: Unknown purpose.
- **DR**: Dosimetry records.
- **SC**: Screenshots.
- **CT**: CT scan data.

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/IotaCZE/kyrykyrybii.git
   cd kyrykyrybii
   ```
1. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
1. **Explore**