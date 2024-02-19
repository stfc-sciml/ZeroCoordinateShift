#!/bin/bash
for i in {1..5}
do
  python main.py --epochs=100 --seed $i > outputs/aligned_baseline_$i.txt
  python main.py --epochs=100 --ZCS --seed $i > outputs/aligned_ZCS_$i.txt
done
