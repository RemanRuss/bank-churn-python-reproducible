# Unit 1 Project Reflection

The original analysis was contained in a single R Markdown file. Data loading, cleaning,
exploratory analysis, model fitting, evaluation, plotting, and written interpretation were mixed
together, and the raw CSV was expected to exist in the working directory. A new user therefore
had to infer both the execution order and the required file locations. Numerical model results
were also typed directly into the narrative, so they could become inconsistent with a rerun.

I transformed the analysis into a Python project with separate stages for preprocessing, model
training, result generation, and report rendering. Reusable logic is now stored in Python
functions, all paths are relative to the repository root, and the active workflow no longer
requires R. The original R Markdown is retained only as an archived record of the starting
analysis. The Python implementation preserves the original four model families and predictor
exclusions while using scikit-learn equivalents.

The largest reproducibility improvements are the project-local `venv`, pinned packages in
`requirements.txt`, the single `make reproduce` command, fixed random seed 123, explicit input
validation, training-only median imputation, and `pytest` tests. KNN and SVM are standardized
inside scikit-learn pipelines so scaling is learned separately within each cross-validation fold.
The generated HTML report reads the current result files, so model AUC values are never hard-coded.

For a future project, I would create the directory structure, environment specification, reusable
functions, and tests at the beginning rather than refactoring a finished analysis. I would also
keep computed quantities connected directly to generated outputs instead of copying numbers into
prose.

**Time spent:**
(1) project structure/environment setup: First time 2-3 minutes, reproduce takes less than 30 seconds
(2) Python refactoring: less than 30 seconds
(3) tests/automation: less than 30 seconds
(4) documentation: less than 30 seconds

