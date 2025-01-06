# README

## Libraries Used

### Task 1
- os
- csv
- argparse
- random
- re
- ast
- json
- time
- tracemalloc
- sklearn (SVC, KNeighborsClassifier, RandomForestClassifier, classification_report, accuracy_score)
- numpy
- collections
- matplotlib

### Task 2
- json
- matplotlib (pyplot)
- matplotlib_venn (venn3)
- argparse
- os

### Task 3
- os
- csv
- argparse
- re
- ast
- json
- sklearn (SVC, KNeighborsClassifier, RandomForestClassifier, RFE, permutation_importance)
- numpy
- collections
- matplotlib (pyplot)

---

## How to Run the Codes

### Task 1
python3 a4task1ab.py --folder ../dataset/packet_size_direction --k 10 --scenario closed --scaling min_max --ensemble p1_p2_diff --n 45

python3 a4task1cd.py --folder ../dataset/packet_size_direction --k 10 --scenario closed --scaling min_max --ensemble p1_p2_diff --n 45

### Task 2
python3 a4task2.py --input_dir ../../task1/output/ --input_file 1a_n10_data.json --output_dir ../output

### Task 3
python3 a4task3.py --folder ../dataset/packet_size_direction --k 10 --scenario closed --scaling min_max --n 45
