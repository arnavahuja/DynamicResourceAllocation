# Optional Terraform: enables required APIs, creates the Artifact Registry
# repo, and creates the GCS bucket for model checkpoints. Cloud Run services
# themselves are deployed by `gcp/deploy.sh` or `gcp/cloudbuild.yaml`, since
# managing them with Terraform tends to fight image-tag pushes.
#
# Usage:
#   cd gcp/terraform
#   terraform init
#   terraform apply -var project_id=<your-project> -var bucket_name=<your-bucket>

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.0" }
  }
}

variable "project_id"  { type = string }
variable "region"      { type = string, default = "us-central1" }
variable "bucket_name" { type = string }

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "cloud_rl" {
  location      = var.region
  repository_id = "cloud-rl"
  description   = "Cloud RL Scheduler images"
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

resource "google_storage_bucket" "checkpoints" {
  name                        = var.bucket_name
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true
  depends_on                  = [google_project_service.apis]

  lifecycle_rule {
    condition { age = 90 }
    action { type = "Delete" }
  }
}

output "artifact_registry_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.cloud_rl.repository_id}"
}

output "bucket_url" {
  value = "gs://${google_storage_bucket.checkpoints.name}"
}
