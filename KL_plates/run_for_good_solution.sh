#!/bin/bash

# this one is too expensive for us to run on 5 seeds
python main.py --seed 0 > outputs/aligned_baseline_solution.txt

# for ZCS, we run on 5 seeds
for i in {1..5}
do
  python main.py --ZCS --seed $i > outputs/aligned_ZCS_solution_$i.txt
done
