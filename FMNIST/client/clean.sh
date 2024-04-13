#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <number>"
    exit 1
fi

number=$1

files=(
"grads.pkl agg_gradients.pkl" "b_cap.pkl" "client_acc_${number}.pkl" "client_dataset_${number}.pkl" "coeff_a.pkl" "coeff_b_${number}.pkl" "creds.pkl" "csum1.pkl" "ds_${number}.pkl" "iris.keras" "local_accuracy_${number}.pkl" "local_gradients_${number}.pkl" "test_data_X.pkl" "test_data_y.pkl"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "File '$file' removed"
    else
        echo "File '$file' not found"
    fi
done

rm quant*
rm R*

folder="tf_ckpts/"
if [ -d "$folder" ]; then
    rm -rf "$folder"
    echo "Folder '$folder' removed"
else
    echo "Folder '$folder' not found"
fi