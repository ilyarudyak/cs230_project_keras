#!/usr/bin/env bash
gcloud compute scp --recurse --zone "us-central1-a" --project "electric-autumn-245617" \
ilyarudyak@instance-2-gpu:~/cs230_project_keras/isbi_challenge/experiments/custom_metric/weights.07* .
