# Netflix Recommendation System

## Problem Description
This project implements a recommendation system utilizing the Netflix Prize Dataset. 
The dataset contains approximately 100M ratings from 480K users across 17.7K movies.

## Setup Instructions
1. Create a virtual environment:
   `python -m venv venv`
2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
3. Install the dependencies:
   `pip install -r requirements.txt`
4. Download the Netflix Prize Dataset from Kaggle and place it in the `data/raw/` directory.

## How to Run Each Phase
- **Data Prep**: run the scripts for data preprocessing in `scripts/`
- **Modeling**: train models using `scripts/` or `notebooks/`
- **Evaluation**: run the evaluation scripts to generate metrics in `reports/`

## Folder Structure Tree
```text
Shubham_project/
├── config.py
├── dashboard/
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
├── README.md
├── reports/
├── requirements.txt
├── scripts/
├── setup.py
└── src/
    ├── __init__.py
    └── models/
```

## Results
| Model       | RMSE  | MAE   | MAP@10 | Coverage |
|-------------|-------|-------|--------|----------|
| KNN (User)  | ~1.02 | ~0.81 | ~0.08  | ~0.12    |
| SVD         | ~0.91 | ~0.72 | ~0.14  | ~0.45    |
| Neural CF   | ~0.94 | ~0.74 | ~0.12  | ~0.38    |

## Dashboard
To launch the interactive recommendation dashboard locally, run:
`streamlit run dashboard/app.py`

## Git History
The development followed a phased approach using 6 specific commits:
1. "Scaffold project structure and configuration files"
2. "Implement data loader and preprocessing scripts"
3. "Add EDA notebook analyzing rating distributions and sparsity"
4. "Implement KNN, SVD, and Neural Collaborative Filtering models"
5. "Add evaluation metrics, recommender engine wrapper, and CLI tools"
6. "Build Streamlit dashboard, add report scaffolds, and update README"

## Known Limitations
- **Cold Start Problem:** New users or items with zero ratings cannot be effectively modeled without content-based fallbacks.
- **Popularity Bias:** Models (especially SVD) naturally bias towards frequently rated mainstream movies.
- **Computational Constraints:** The sheer size of the dataset requires sampling for manageable iteration speeds.
- **Temporal Drift:** User preferences change over time, which isn't dynamically modeled here.
