.PHONY: setup data train evaluate compare-baselines report serve dashboard docker all clean

# ============================================================
# Spot Instance RL Optimizer - Makefile
# ============================================================

# --- Setup ---
setup:
	pip install -r requirements.txt

# --- Data Pipeline ---
data-synthetic:
	cd data/scripts && python generate_synthetic_spot_prices.py \
		--region ap-southeast-1 \
		--instance-types m5.large \
		--days 30 \
		--volatility 0.10 \
		--spike-prob 0.005 \
		--spike-multiplier 1.5 \
		--tag stable \
		--output ../raw/spot_prices/
	cd data/scripts && python generate_synthetic_spot_prices.py \
		--region ap-southeast-1 \
		--instance-types m5.large \
		--days 30 \
		--volatility 0.25 \
		--spike-prob 0.05 \
		--spike-multiplier 4.0 \
		--tag volatile \
		--output ../raw/spot_prices/
	cd data/scripts && python generate_synthetic_spot_prices.py \
		--region ap-southeast-1 \
		--instance-types m5.large \
		--days 30 \
		--volatility 0.18 \
		--spike-prob 0.08 \
		--spike-multiplier 5.0 \
		--tag spike \
		--output ../raw/spot_prices/

data-workload:
	cd data/scripts && python generate_workload.py \
		--duration 30 \
		--pattern batch_ml_training \
		--output ../raw/workload.csv

data-aws:
	cd data/scripts && python fetch_spot_prices.py \
		--region ap-southeast-1 \
		--instance-types m5.large \
		--days 30 \
		--output ../raw/spot_prices/

data-preprocess:
	cd data/scripts && python preprocess.py \
		--input ../raw/ \
		--input-glob "spot_prices/*stable*.csv" \
		--output ../processed/ \
		--output-name price_features_stable \
		--instance-type m5.large
	cd data/scripts && python preprocess.py \
		--input ../raw/ \
		--input-glob "spot_prices/*volatile*.csv" \
		--output ../processed/ \
		--output-name price_features_volatile \
		--instance-type m5.large
	cd data/scripts && python preprocess.py \
		--input ../raw/ \
		--input-glob "spot_prices/*spike*.csv" \
		--output ../processed/ \
		--output-name price_features_spike \
		--instance-type m5.large

data: data-synthetic data-workload data-preprocess

# --- Training ---
train:
	python experiments/train.py \
		--config experiments/configs/dqn_default.yaml \
		--experiment-name dqn_default

train-quick:
	python experiments/train.py \
		--config experiments/configs/dqn_quick_test.yaml \
		--experiment-name quick_test

# --- Evaluation ---
evaluate:
	python experiments/evaluate.py \
		--config experiments/configs/stable_price.yaml \
		--model results/models/dqn_default_best.pth \
		--episodes 50

compare-baselines:
	python experiments/compare_baselines.py \
		--scenarios stable,volatile,spike \
		--dqn-model results/models/dqn_default_best.pth \
		--episodes 50

# --- Serving ---
serve:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard.py

mlflow-ui:
	mlflow ui --port 5000

# --- Docker ---
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# --- Full Pipeline ---
all: data train evaluate compare-baselines report
	@echo "Full pipeline complete!"
	@echo "  - Data: data/raw/ and data/processed/"
	@echo "  - Model: results/models/dqn_default_best.pth"
	@echo "  - Evaluation: results/reports/"
	@echo ""
	@echo "Next steps:"
	@echo "  make serve      # Start API at http://localhost:8000"
	@echo "  make dashboard  # Start dashboard at http://localhost:8501"

report:
	python experiments/generate_report.py --results-dir results

# --- Clean ---
clean:
	rm -rf results/models/*.pth
	rm -rf results/plots/*
	rm -rf results/logs/*
	rm -rf results/reports/*
	rm -rf mlruns/
	rm -rf __pycache__
	find . -name "__pycache__" -type d -exec rm -rf {} +
