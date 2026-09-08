.PHONY: setup data baseline train train-smoke spark-eda error-analysis kaggle-notebook calibrate probe demo test clean

CONFIG ?= config/config.yaml

setup:
	pip install -r requirements.txt

# Downloads train.csv via the Kaggle API into data/raw/. Requires
# ~/.kaggle/kaggle.json (see README "Getting the data").
data:
	kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification -p data/raw
	cd data/raw && unzip -o jigsaw-unintended-bias-in-toxicity-classification.zip

# Stage 0: TF-IDF + LogisticRegression baseline. CPU, ~20 min on the full
# dataset. Pass SAMPLE=50000 to iterate faster.
baseline:
	python src/baseline.py $(if $(SAMPLE),--sample $(SAMPLE),) --save-model

# Stage 1/2: DeBERTa multi-task fine-tune. Meant to run where a GPU is
# available (Kaggle/Colab) -- see README for timing. Pass SAMPLE=2000 for
# a CPU structural smoke test only (no real result).
train:
	python src/train.py --config $(CONFIG) $(if $(SAMPLE),--sample $(SAMPLE),)

train-smoke:
	$(MAKE) train SAMPLE=2000

# Regenerate notebooks/kaggle/train_deberta_kaggle.ipynb from the current
# src/ files -- run this after any change to metrics.py/data.py/model.py/
# train.py so the Kaggle notebook never drifts from what's tested locally.
kaggle-notebook:
	python scripts/gen_kaggle_notebook.py

# Executes the Spark EDA notebook in place (writes outputs back into the
# .ipynb, plus reports/figures/ and data/processed/*.parquet). Needs a
# JVM on PATH -- see README "Where Spark fits" for the Windows/winutils.exe
# setup this project's dev machine required.
spark-eda:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_spark.ipynb

error-analysis:
	jupyter nbconvert --to notebook --execute --inplace notebooks/03_error_analysis.ipynb

# Standalone module self-tests (each src/*.py has a `python src/x.py`
# smoke test using synthetic data -- run all of them; no dataset needed).
test:
	python src/metrics.py
	python src/data.py
	python src/calibrate.py
	python src/thresholds.py
	python src/bias_probe.py
	python src/infer.py
	python src/error_analysis.py

calibrate:
	python src/calibrate.py

probe:
	python src/bias_probe.py

demo:
	uvicorn app.main:app --reload --port 8000

clean:
	rm -rf reports/figures/*.png reports/*.csv
