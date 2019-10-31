#!/usr/bin/env bash
gcloud compute scp --recurse --zone "us-central1-a" --project "electric-autumn-245617" \
ilyarudyak@instance-2-gpu:~/cs230_project_keras/isbi_challenge/experiments/augmentation/history_crop_size_32.pickle .
#ilyarudyak@instance-2-gpu:~/cs230_project_keras/isbi_challenge/experiments/lr_tuning/weig* \
#ilyarudyak@instance-2-gpu:~/cs230_project_keras/isbi_challenge/experiments/lr_tuning/even* \
#.
