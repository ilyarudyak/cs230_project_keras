#!/usr/bin/env bash
gcloud compute scp --recurse --zone "us-central1-a" --project "electric-autumn-245617" \
ilyarudyak@instance-2-gpu:~/cs230_project_keras/isbi_challenge/experiments/updated_unet/ev* .
