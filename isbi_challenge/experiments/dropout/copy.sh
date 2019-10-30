#!/usr/bin/env bash
gcloud compute scp --recurse --zone "us-central1-a" --project "electric-autumn-245617" \
ilyarudyak@instance-2-gpu:~/cs230_project_keras/isbi_challenge/experiments/dropout/hist* .
#ilyarudyak@instance-2-gpu:~/cs230_project_keras/isbi_challenge/experiments/dropout/weig* \
#ilyarudyak@instance-2-gpu:~/cs230_project_keras/isbi_challenge/experiments/dropout/even* \
#.
