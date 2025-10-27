## EEG-Based Classification in Psychiatry Using Motif Discovery

This repository contains code and notebooks for EEG signal preprocessing, motif discovery, feature extraction, and classification using various machine learning models. The workflow is organized into two main stages:

### Project Structure

1. **Pre-processing** – Cleaning EEG signals and extracting frequency bands.
2. **Main Code** – Motif discovery, feature computation, and classification.

---

#### Pre-processing

Located in the `pre-processing/` folder.

- **Preprocessing [DATASET_NAME].ipynb**     
  Performs dataset-specific preprocessing and cleaning of EEG signals.
  Outputs are stored as pickle files for later use.

- **Band Extraction.ipynb**  
  Uses the [MNE](https://mne.tools/stable/index.html) library to extract **alpha**, **beta**, and **theta** frequency bands from EEG signals. The extracted bands are stored in separate pickle files.

---

####  Main Code

Located in the `main-code/` folder.

##### Motif Discovery
- **All-electrodes-bands-motif-discovery SCRIMP++.ipynb**  
  Extracts motifs using the **SCRIMP++** algorithm (older approach).

- **All-electrodes-bands-motif-discovery k-Motiflets.ipynb**  
  Implements motif discovery using the **k-Motiflets** algorithm.

  The motif discovery notebooks read the pickle files with the help of the *utils.py* helper functions and as input takes into account each band separately. The output of these notebooks is the feature matrix, computed based on the top ranked extracted motifs (using the difference score). The output feature matrix is separated to train, validation and testing CSV files. 

##### Baseline approach
- **TSFRESH_feature_computation.ipynb**  
  Computes baseline feature matrices using the [TSFRESH](https://tsfresh.readthedocs.io/) library.

##### Classification
- **Classification-Model-tunning-Experiments.ipynb**  
  Explores hyperparameters using the `hyperopt` library to identify best hyperparameters. 
  Models tested include:
  - `SVC` (RBF kernel)
  - `LinearSVC`
  - `RandomForestClassifier`
  - `DecisionTreeClassifier`
  - `LogisticRegression`
  - `MLPClassifier`  
  *(Training and validation sets only)*

- **Classification-Evaluation-CV-5fold.ipynb**  
  Performs **5-fold cross-validation** to compute final testing results for each model.

---

#### How to Use

1. **Preprocess EEG data**  
   Dataset-specific cleaning and removing damaged recordings, see `pre-processing/` for examples.

2. **Extract frequency bands**  
   Use `Band Extraction.ipynb` to obtain alpha, beta, and theta bands.

3. **Motif discovery & feature computation**  
   Run motif discovery notebooks in `main-code/` and compute features.

4. **Train & evaluate models**  
   Use classification notebooks to tune hyperparameters and evaluate performance.

---

#### Requirements

- Python 3.9
- [MNE](https://mne.tools/stable/index.html)
- [TSFRESH](https://tsfresh.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [hyperopt](https://github.com/hyperopt/hyperopt)
- [stumpy](https://stumpy.readthedocs.io/) 
- [k-Motiflets](https://github.com/motiflets/k-Motiflets)
- numpy, pandas, matplotlib


