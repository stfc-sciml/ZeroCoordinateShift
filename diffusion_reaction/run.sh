#!/bin/bash
for i in {1..5}
do
  python main.py --seed $i > outputs/aligned_baseline_$i.txt
  python main.py --ZCS --seed $i > outputs/aligned_ZCS_$i.txt
  python main.py --unaligned --seed $i > outputs/unaligned_baseline_$i.txt
done
