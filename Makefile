.PHONY: setup test demo serve run-public run-private sweep validate check hud bank priors sft train clean
PY ?= python
DATA ?= data
MANIFEST ?= $(DATA)/manifest.json
PRIVATE ?= $(DATA)/private
LEVELS ?= 1,2,3
OUT ?= submissions/run_$(shell date +%H%M%S).json

setup:            ## core deps (CPU) — add [gate] / [serve] / [train] extras on GPU boxes
	pip install -e ".[dev]"

test:             ## unit + e2e (mock) tests
	$(PY) -m pytest -q

demo:             ## full cascade on synthetic clips with mock models, then open the HUD
	$(PY) -c "from tests.synth import make_video as m; import os; os.makedirs('data/demo',exist_ok=True); \
	m('data/demo/D01.mp4',20); m('data/demo/D02.mp4',60,fire_windows=[(20,35)]); m('data/demo/D03.mp4',90); m('data/demo/D04.mp4',120,fire_windows=[(10,30),(80,100)])"
	printf 'video_id,level,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary\nD01,1,0,normal,,,\nD02,2,1,fire,20,35,\nD03,3,0,normal,,,\nD04,3,1,fire,10,30,\nD04,3,1,fire,80,100,\n' > data/demo/ground_truth.csv
	$(PY) -m sentinel.cli run --gt data/demo/ground_truth.csv --videos data/demo --out submissions/demo.json --mock
	cp submissions/demo.trace.json dashboard/demo.trace.json; cp data/demo/ground_truth.csv dashboard/demo_gt.csv
	@echo "open http://localhost:8765/index.html?trace=demo.trace.json&gt=demo_gt.csv"; cd dashboard && $(PY) -m http.server 8765

serve:            ## start the verifier VLM (vLLM). MODEL=... LORA=... PORT=...
	bash scripts/serve.sh vllm

bank:             ## SigLIP2 normality bank from train/normal
	$(PY) -m sentinel.cli build-normal-bank --train-dir $(DATA)/train

priors:           ## per-class median durations from train GT
	$(PY) -m sentinel.cli derive-priors --train-dir $(DATA)/train

run-public:       ## run + score on the public test set (calibration)
	$(PY) -m sentinel.cli run --gt $(DATA)/test/ground_truth.csv --videos $(DATA)/test/videos --out submissions/public.json

sweep:            ## decoder-only threshold sweep from the public trace (seconds)
	$(PY) -m sentinel.cli sweep --gt $(DATA)/test/ground_truth.csv --videos $(DATA)/test/videos --trace-from submissions/public.trace.json

run-private:      ## run on the arena manifest, chosen LEVELS
	$(PY) -m sentinel.cli run --manifest $(MANIFEST) --videos $(PRIVATE) --levels $(LEVELS) --out $(OUT)

validate:         ## arena-style validation. FILE=...
	$(PY) -m sentinel.cli validate $(FILE) --manifest $(MANIFEST)

check:            ## upload gate. FILE=...
	$(PY) -m sentinel.cli submit-check $(FILE) --manifest $(MANIFEST) --public submissions/public.json

hud:              ## serve the dashboard for any trace
	cd dashboard && $(PY) -m http.server 8765

sft:              ## build LoRA dataset
	$(PY) scripts/make_sft_dataset.py --train-dir $(DATA)/train --out $(DATA)/sft

train:            ## LoRA (GPU)
	$(PY) scripts/train_lora_unsloth.py --data $(DATA)/sft --out output/qwen3vl4b-sentinel-lora

clean:
	rm -rf cache/*.npz .pytest_cache

app:              ## live application (real models per configs/default.yaml; needs `make serve` running)
	uvicorn app.server:app --host 0.0.0.0 --port 8080

app-mock:         ## live application with CPU mock models (UI/flow identical)
	SENTINEL_MOCK=1 uvicorn app.server:app --host 0.0.0.0 --port 8080
