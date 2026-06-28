# polars-lm

This repo now contains a small decoder-only language model that can be trained on the synthetic Polars dataset in `data/train.txt` and then run locally or on Google Cloud Vertex AI.

## Local training

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py --data-path data/train.txt --output-dir artifacts
```

Generate from a saved checkpoint:

```bash
python generate.py --model-path artifacts/model.pt --vocab-path artifacts/vocab.json
```

## Google Cloud Vertex AI

1. Create a GCS bucket for training inputs and outputs.
2. Upload `data/train.txt` to GCS.
3. Build the container from this repo.
4. Submit a custom training job on Vertex AI using the Docker image from this repo.

Example container build and push:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
gcloud artifacts repositories create polars-lm --repository-format=docker --location=us-central1
gcloud builds submit --tag us-central1-docker.pkg.dev/PROJECT_ID/polars-lm/trainer:latest
```

Upload the dataset:

```bash
gcloud storage cp data/train.txt gs://YOUR_BUCKET/datasets/train.txt
```

Example job submission:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=polars-lm-train \
  --worker-pool-spec=machine-type=n1-standard-8,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=us-central1-docker.pkg.dev/PROJECT_ID/polars-lm/trainer:latest \
  --args=--data-path,/gcs/YOUR_BUCKET/datasets/train.txt,--output-dir,/gcs/YOUR_BUCKET/polars-lm-artifacts,--epochs,5,--batch-size,64
```

Notes:

- The repo includes `vertex_job.yaml` if you prefer `gcloud ai custom-jobs create --region=us-central1 --config=vertex_job.yaml`.
- Vertex AI reads the dataset and writes checkpoints through the `/gcs/YOUR_BUCKET/...` path.
- Start with a T4 GPU. Move to A100 only if training time is the bottleneck.

## Files

- `train.py`: training entrypoint
- `model.py`: decoder-only transformer
- `dataset.py`: character vocabulary and next-token dataset
- `generate.py`: inference script
