#!/bin/bash
for i in {1..5}
do
  python main.py --epochs=1000 --seed $i > outputs/aligned_baseline_$i.txt
  python main.py --epochs=1000 --ZCS --seed $i > outputs/aligned_ZCS_$i.txt
  python main.py --epochs=1000 --unaligned --seed $i > outputs/unaligned_baseline_$i.txt
done
