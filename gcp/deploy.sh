#!/usr/bin/env bash
# Deploy the cloud-rl-scheduler stack to GCP Cloud Run.
#
# Reads from the repo's .env: GCP_PROJECT_ID, GCP_REGION, GCP_BUCKET_NAME.
# Requires `gcloud` authenticated (`gcloud auth login`) and the project's
# Artifact Registry + Cloud Run + Cloud Storage APIs enabled.
#
# Usage:
#   bash gcp/deploy.sh [--build-only] [--backend-only] [--frontend-only]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Load .env ────────────────────────────────────────────────────
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID in .env}"
: "${GCP_REGION:=us-central1}"
: "${GCP_BUCKET_NAME:?Set GCP_BUCKET_NAME in .env}"

REPO_NAME="cloud-rl"
BACKEND_IMG="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO_NAME}/backend:latest"
FRONTEND_IMG="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO_NAME}/frontend:latest"

BUILD_ONLY=false
BACKEND_ONLY=false
FRONTEND_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --build-only)    BUILD_ONLY=true ;;
        --backend-only)  BACKEND_ONLY=true ;;
        --frontend-only) FRONTEND_ONLY=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "▶ Project: $GCP_PROJECT_ID  Region: $GCP_REGION  Bucket: $GCP_BUCKET_NAME"

# ── 0. One-time setup (idempotent) ───────────────────────────────
echo "▶ Ensuring Artifact Registry repo exists..."
gcloud artifacts repositories describe "$REPO_NAME" \
    --location "$GCP_REGION" --project "$GCP_PROJECT_ID" >/dev/null 2>&1 || \
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$GCP_REGION" \
        --description="Cloud RL Scheduler images" \
        --project "$GCP_PROJECT_ID"

echo "▶ Ensuring GCS bucket exists..."
gcloud storage buckets describe "gs://${GCP_BUCKET_NAME}" \
    --project "$GCP_PROJECT_ID" >/dev/null 2>&1 || \
    gcloud storage buckets create "gs://${GCP_BUCKET_NAME}" \
        --project "$GCP_PROJECT_ID" --location "$GCP_REGION"

gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" -q

# ── 1. Build & push images ───────────────────────────────────────
if ! $FRONTEND_ONLY; then
    echo "▶ Building backend image..."
    docker build -f Dockerfile.backend -t "$BACKEND_IMG" .
    echo "▶ Pushing backend image..."
    docker push "$BACKEND_IMG"
fi

if ! $BACKEND_ONLY; then
    echo "▶ Building frontend image..."
    docker build -f Dockerfile.frontend -t "$FRONTEND_IMG" .
    echo "▶ Pushing frontend image..."
    docker push "$FRONTEND_IMG"
fi

if $BUILD_ONLY; then
    echo "✔ Build complete. Skipping deploy."
    exit 0
fi

# ── 2. Deploy backend (Cloud Run) ────────────────────────────────
if ! $FRONTEND_ONLY; then
    echo "▶ Deploying backend to Cloud Run..."
    gcloud run deploy cloud-rl-backend \
        --image "$BACKEND_IMG" \
        --region "$GCP_REGION" \
        --project "$GCP_PROJECT_ID" \
        --platform managed \
        --allow-unauthenticated \
        --min-instances 1 \
        --max-instances 4 \
        --cpu 2 --memory 4Gi \
        --timeout 3600 \
        --set-env-vars "GCP_PROJECT_ID=${GCP_PROJECT_ID},GCP_BUCKET_NAME=${GCP_BUCKET_NAME},N_SERVERS=${N_SERVERS:-10}"

    BACKEND_URL=$(gcloud run services describe cloud-rl-backend \
        --region "$GCP_REGION" --project "$GCP_PROJECT_ID" \
        --format 'value(status.url)')
    echo "✔ Backend: $BACKEND_URL"
fi

# ── 3. Deploy frontend (Cloud Run, nginx-served SPA) ─────────────
if ! $BACKEND_ONLY; then
    echo "▶ Deploying frontend to Cloud Run..."
    gcloud run deploy cloud-rl-frontend \
        --image "$FRONTEND_IMG" \
        --region "$GCP_REGION" \
        --project "$GCP_PROJECT_ID" \
        --platform managed \
        --allow-unauthenticated \
        --min-instances 0 \
        --max-instances 4 \
        --cpu 1 --memory 256Mi

    FRONTEND_URL=$(gcloud run services describe cloud-rl-frontend \
        --region "$GCP_REGION" --project "$GCP_PROJECT_ID" \
        --format 'value(status.url)')
    echo "✔ Frontend: $FRONTEND_URL"
fi

echo "▶ Done."
