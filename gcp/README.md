# Cloud RL Scheduler — Deployment

Two paths: **Docker Compose** (local containers) and **GCP Cloud Run**
(production). Neither is required to develop locally — `uvicorn` + `npm
run dev` works without either.

## Local containers (Docker Compose)

```bash
cp .env.example .env       # set N_SERVERS etc.
docker compose up --build  # http://localhost:5173
```

The frontend container nginx-serves the built SPA on `:5173`. The backend
runs on `:8000`. SQLite lives in `./data/experiments.db`. Checkpoints land
in `./checkpoints/`.

## GCP Cloud Run (production)

Prereqs:
- A GCP project with billing enabled.
- `gcloud` CLI installed and authenticated (`gcloud auth login`,
  `gcloud config set project <id>`).
- `docker` running locally (Cloud Build also works — see below).

One-time:
```bash
# .env must contain:
#   GCP_PROJECT_ID=<your-project>
#   GCP_REGION=us-central1
#   GCP_BUCKET_NAME=<unique-bucket-name>

# Optional infra-as-code (creates Artifact Registry + GCS bucket):
cd gcp/terraform
terraform init
terraform apply -var project_id=<your-project> -var bucket_name=<your-bucket>
cd ../..
```

Deploy:
```bash
bash gcp/deploy.sh
```

Flags:
- `--build-only` — build & push images, skip deploy.
- `--backend-only` / `--frontend-only` — partial deploy.

The script:
1. Ensures Artifact Registry repo `cloud-rl` exists in `$GCP_REGION`.
2. Ensures GCS bucket `$GCP_BUCKET_NAME` exists.
3. Builds and pushes `backend` + `frontend` images.
4. `gcloud run deploy`s both as managed services with public ingress.
   Backend: 2 vCPU / 4 GiB / min-1 instance / 60-min request timeout.
   Frontend: 1 vCPU / 256 MiB / min-0 instance.

Trained model checkpoints are uploaded to
`gs://$GCP_BUCKET_NAME/runs/<run_id>/{best,last}.pt` automatically when
training completes (the backend silently no-ops if `GCP_PROJECT_ID` is
unset, which is what local dev relies on).

## Cloud Build (CI)

Connect this repo to a Cloud Build trigger pointed at
`gcp/cloudbuild.yaml`. Substitution variables to set:
- `_REGION` (e.g. `us-central1`)
- `_BUCKET_NAME` (must match an existing bucket)
- `_N_SERVERS` (default cluster size baked into backend env)

Push to `main` → both services rebuild and redeploy automatically.

## Same-origin routing (production)

Cloud Run gives each service its own URL by default. To put both behind
one origin (so the frontend's relative `/api` and `/ws` paths work
without CORS), use a Google Cloud Load Balancer with two backends:
- `/api/*` and `/ws/*` → `cloud-rl-backend`
- everything else → `cloud-rl-frontend`

Until the LB is set up, the simplest workaround is to set
`VITE_API_URL=https://cloud-rl-backend-...run.app` at frontend build
time and update `frontend/src/lib/api.js` to read it. (Not wired by
default — local dev uses Vite's proxy, which is preferable.)
