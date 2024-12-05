# README

## Libraries Used

### Task 1
- os
- csv
- argparse
- random
- re
- sklearn (SVC, KNeighborsClassifier, RandomForestClassifier, classification_report)
- numpy

### Task 2
- os
- argparse
- numpy
- pandas
- tensorflow (Sequential, Conv1D, BatchNormalization, ReLU, Dropout, Dense, Flatten, Input, Activation, MaxPooling1D)
- sklearn (train_test_split, confusion_matrix)

### Task 3
- argparse
- os
- sys
- importlib

---

## How to Run the Codes

### Task 1
python3 a3task1.py --folder ../dataset/inter_arrival_time --k 4 --scenario closed --scaling min_max --ensemble p1_p2_diff

### Task 2
python3 a3task2.py --main_folder ../input_pairs --n 80 --m 48 --include_features

### Task 3

#### Traditional Machine Learning
python3 a3task3.py -s traditional_ml --folder ../../task1/dataset/packet_size_direction --k 4 --scenario closed --scaling z_score --ensemble p1_p2_diff

#### Deep Learning
python3 a3task3.py -s deep_learning --main_folder ../../task2/input_pairs --n 80 --m 40 --include_features

#### Packet Analysis
python3 a3task3.py -s packet_analysis --base_directory ../../../Assignment_2/task1/pcap --output_file ../../../Assignment_2/task1/output/task_1a_analysis_results.txt

#### Statistical Analysis
python3 a3task3.py -s statistical_analysis --base_directory ../../../Assignment_2/task1 --scenario 1

#### Synthetic Data Generation
python3 a3task3.py -s synthetic_data --data_file ../../../Assignment_2/task2/dataset/csv/packet_size/scenario1_packet_size.csv --synthetic_percentage 20 --k_values 3 --normalization min_max --output_path ../../../Assignment_2/task2/output/

#### Manual Feature Generation
python3 a3task3.py -s manual_feature_gen --base_directory ../../../Assignment_2/task3/dataset/packet_size_and_direction --file_index 1 --m 10

#### Flow to CSV
python3 a3task3.py -s flow_to_csv --base_directory ../../../Assignment_1 --scenario 1

#### Calculate Euclidean Distance
python3 a3task3.py -s calculate_euclidean --scenario scenario_6 --base_path ../../../Assignment_1/output

#### Generate Histogram
python3 a3task3.py -s generate_histogram --base_path ../../../Assignment_1/output
