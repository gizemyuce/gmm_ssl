#!/bin/bash

# for data in a9a vehicleNorm webdata_wXa sylva_prior jasmine madeline philippine musk; do
for data in a9a vehicleNorm jasmine madeline philippine musk mnist_5v9 mnist_5v6 mnist_3v8 mnist_1v7 mnist_0v1; do
  echo ${data}
  python3.9 script_ssl_comparison.py --dataset_name=${data} &> logs/${data}.log &
done

