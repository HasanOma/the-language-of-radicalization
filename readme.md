# The Language of Radicalization: A Cross-Ideological NLP Study

This repository contains the full data processing and analysis pipeline for the Master's Thesis, "The Language of Radicalization: A Cross-Ideological Natural Language Processing Study of Extremist Rhetoric on Social Media," submitted to the Norwegian University of Science and Technology (NTNU) in June 2025.

## 📖 Project Overview

This research project uses Natural Language Processing (NLP) and Machine Learning (ML) techniques to analyze the linguistic and emotional characteristics of various online extremist groups. The primary questions adressed are:
1.  To identify the dominant emotional cues present in their discourse.
2.  To develop a comparative framework for analyzing similarities and differences in emotional expression across ideologies.
3.  To evaluate the efficacy of state-of-the-art AI models in classifying extremist discourse based on its linguistic features.

The project involves a multi-stage pipeline, from ingesting raw data from forums and social media to training and evaluating deep learning models for ideology classification.


## 📂 Repository Structure

The project is organized into a modular pipeline, with distinct scripts for preprocessing and for answering each of the three main research questions.

```text
.
├── data/                         # Created automatically by scripts
│   └── revised-preprocess/       # Contains subfolders for each dataset size variant
│       └── data80k/
│           ├── 01_raw.feather    # Raw posts/messages after initial ingestion
│           └── 02_features.feather # Cleaned text + engineered features
├── results-revised/              # Created automatically; all plots, tables, and metrics
│   ├── rq1/
│   ├── rq2/
│   └── rq3/
├── Colab_notebook.ipynb          # Notebook with example execution commands for Google Colab
├── config_ideology.json          # 🔹 Maps forums/channels to their ideology class
├── preprocess.py                 # 🔹 STEP 1: Ingests, cleans, and engineers features
├── rq1_emotion_analysis.py       # 🔸 STEP 2a: RQ1, Dominant emotions analysis
├── rq2_similarity_stats.py       # 🔸 STEP 2b: RQ2, Cross-group similarity analysis
├── rq3_classification.py         # 🔸 STEP 2c: RQ3, Ideology text classification
└── requirements.txt              # Python package dependencies
```

## ⚙️ Setup and Installation

### Requirements
* Python 3.10+ (The thesis experiments were run on Python 3.11)
* A running PostgreSQL server
* Access to a CUDA-enabled GPU is **highly recommended** for running the transformer model experiments (RQ3). The final models were trained on an NVIDIA A100 GPU.

### Step-by-Step Installation

1.  **Clone the Repository:**
```bash
    git clone https://github.com/HasanOma/the-language-of-radicalization.git
    cd the-language-of-radicalization
```

2.  **Create and Activate a Python Virtual Environment (Recommended):**
```bash
    python -m venv .venv
    source .venv/bin/activate  
    # On Windows: .venv\Scripts\activate
```

3.  **Install Dependencies:**
    A `requirements.txt` is placed in the root directory, created based on the necessary packages.
```bash
    pip install requirements.txt
```
`Colab_notebook.ipynb` Contains the entire script one needs if wishing to run the experiments solely on Google Colab.

4.  **Download NLP Models:**
    The preprocessing script uses a spaCy model. Download it via:
```bash
    python -m spacy download en_core_web_sm
```
NLTK data (e.g., 'stopwords', 'wordnet') will be downloaded automatically by `preprocess.py` on its first run if not found.


## 🗃️ Data Preparation

This research relies on both public and restricted-access datasets. **You must acquire this data yourself and set up the database.**

1.  **Public Dataset (Kaggle):**
    * Download the "How ISIS Uses Twitter" dataset from Kaggle: [https://www.kaggle.com/datasets/fifthtribe/how-isis-uses-twitter](https://www.kaggle.com/datasets/fifthtribe/how-isis-uses-twitter)
    * Place the `tweets.csv` file in the root directory of this project.

2.  **Restricted Datasets (Cambridge Cybercrime Centre):**
    * The primary data sources for this research are the **CrimeBB** and **ExtremeBB** datasets.
    * **These datasets are restricted and are not included in this repository (that is why the preprocessed data is not available as well).** Access must be formally requested and approved by the **Cambridge Cybercrime Centre (CCC)** at the University of Cambridge.

3.  **Database Setup:**
    * The `preprocess.py` script expects the CCC datasets (CrimeBB, ExtremeBB) to be imported into a **PostgreSQL** database.
    * You must set up a PostgreSQL server and import the data dumps according to the schema provided by the CCC.
    * Set the following environment variables to allow the script to connect to your database. **Do not commit these credentials to version control.**

```bash
    export PSQL_DB="your_database_name"
    export PSQL_USER="your_username"
    export PSQL_PWD="your_password"
    export PSQL_HOST="localhost"  # Or your DB host
    export PSQL_PORT="5432"       # Or your DB port
```


## 🚀 Running the Experiments

The workflow must be executed in order. **The `preprocess.py` script must be run first.**

### Step 1: Preprocessing the Data

This script reads from your PostgreSQL database and the `tweets.csv` file, processes all the data (cleaning, feature engineering), and saves the final feature set to a `.feather` file inside a `data/` subdirectory.

To generate a specific dataset variant (e.g., the 80k variant), first ensure the `LIMIT_PER_FORUM` and `LIMIT_PER_SOCIAL` variables inside `preprocess.py` are set to `80000`. Then run:
```bash
    python preprocess.py
```

Note: This is a long-running and memory-intensive process, potentially taking several hours depending on your hardware and the size of the dataset you wish processed.

### Step 2: Running the Analysis Scripts

Once preprocessing is complete, you can run the scripts for each research question. These scripts are independent of each other but must be edited to point to the correct feature file path (e.g., `data/revised-preprocess/data80k/02_features.feather`).

* **RQ1 - Dominant Emotions Analysis:**
```bash
    python rq1_emotion_analysis.py
```
* **RQ2 - Inter-Ideology Similarity Analysis:**
```bash
    python rq2_similarity_stats.py
```
* **RQ3 - Ideology Classification:**
    This script is computationally intensive. It is highly recommended to run it in a GPU environment like Google Colab (see and use `Colab_notebook.ipynb` for setup commands).

    * To run **only the baseline models** (SVM and Random Forest) for the 10k dataset:
    ```bash
        python rq3_classification.py --data "data/revised-preprocess/data10k/02_features.feather" --model none --run-name "10k/classic_models"
    ```

    * To fine-tune a **transformer model** (e.g., DeBERTa on the 10k dataset):
    ```bash
        python rq3_classification.py \
          --data "data/revised-preprocess/data10k/02_features.feather" \
          --model microsoft/deberta-v3-base \
          --epochs 8 \
          --batch 32 \
          --grad-accum 2 \
          --no-baselines \
          --run-name "10k/deberta_final_run"
    ```

Note: For the thesis all these scripts were ran on Google Colab A100 GPU. Import and use the `Colab_notebook.ipynb` for setup commands to replicate correctly.

## 🔧 Configuration and Customization

Several key parameters can be adjusted to change the scope of the experiments.

| Parameter | Location | Purpose |
| :--- | :--- | :--- |
| **`LIMIT_PER_FORUM`** & **`LIMIT_PER_SOCIAL`** | `preprocess.py` | Sets the maximum number of posts to sample per forum/channel per year. This controls the dataset size variants (5k, 10k, 40k, 80k). Set to `None` for the full dataset. |
| **`DATA`** & `OUT` | `rq*_..._.py` scripts | Variables inside `rq1_emotion_analysis.py` and `rq2_similarity_stats.py` to specify the path to the `02_features.feather` input file. And the wanted placement of the output file. |
| **`--model`** | `rq3_classification.py` | Command-line argument to specify the Hugging Face model ID for fine-tuning. Use `"none"` to only run baselines. |
| **`--epochs`, `--batch`, `--lr`, `--grad-accum`** | `rq3_classification.py` | Command-line arguments to control transformer training hyperparameters. |
| **`config_ideology.json`** | Root directory | A JSON file mapping source names (forums, channels) to their corresponding ideology labels. You can edit this file to add new sources or change mappings based on tha data available in the DB before running `preprocess.py`. |


## 🎓 Citation

If you use the methodology, code, or findings from this thesis in your research, please cite it as follows:

```bibtex
@mastersthesis{Omarzae2025LanguageOfRadicalization,
  author    = {Hasan Rehman Omarzae},
  title     = {{The Language of Radicalization: A Cross-Ideological Natural Language Processing Study of Extremist Rhetoric on Social Media}},
  school    = {Norwegian University of Science and Technology (NTNU)},
  year      = {2025},
  month     = {june},
  address   = {Trondheim, Norway},
  type      = {Master's Thesis}
}
```