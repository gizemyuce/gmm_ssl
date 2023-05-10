#!/bin/bash

# for data in a9a vehicleNorm webdata_wXa sylva_prior jasmine madeline philippine musk; do
for data in vehicleNorm jasmine madeline philippine musk; do
# for data in a9a madeline philippine musk; do
  echo ${data}
  python3.9 script_ssl_comparison.py --dataset_name=${data} &> logs/${data}.log &
done

