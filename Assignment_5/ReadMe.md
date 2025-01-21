# README

## Libraries Used

### Task 1
- os
- re
- ast
- argparse
- numpy
- pandas
- tensorflow
- matplotlib.pyplot
- random
- tensorflow (Sequential, Conv1D, BatchNormalization, Dropout, Dense, Flatten, Input, Activation, MaxPooling1D)
- collections (defaultdict)
- sklearn (SVC, KNeighborsClassifier, RandomForestClassifier, accuracy_score, GridSearchCV)


### Task 2
- os
- re
- ast
- argparse
- numpy
- pandas
- tensorflow
- matplotlib.pyplot
- random
- tensorflow (Sequential, Conv1D, BatchNormalization, Dropout, Dense, Flatten, Input, Activation, MaxPooling1D)
- collections (defaultdict)
- sklearn (SVC, KNeighborsClassifier, RandomForestClassifier, precision_score, recall_score, GridSearchCV)


### Task 3
- os
- argparse
- sys
- importlib (import_module)

---

## How to Run the Codes

### Task 1
python3 a5task1ab.py --main_folder ../dataset/packet_size_direction

python3 a5task1c.py --main_folder ../dataset/packet_size_direction

python3 a5task1de.py --folder ../dataset/packet_size_direction --k 10 --scaling min_max --ensemble p1_p2_diff --n 45 --m 90

python3 grapha5task1de.py

### Task 2
python3 a5task2a.py --folder ../dataset/packet_size_direction --k 5 --scaling min_max --ensemble p1_p2_diff --n 45 --m 150

python3 a5task2b.py --folder ../dataset/packet_size_direction --k 5 --scaling min_max --ensemble p1_p2_diff --n 45 --m 150

python3 a5task2c.py --main_folder ../dataset/packet_size_direction

python3 a5task2d.py --main_folder ../dataset/packet_size_direction

### Task 3

#### Traditional Machine Learning
python3 a5task3.py -s traditional_ml --folder ../../../Assignment_3/task1/dataset/packet_size_direction --k 4 --scenario closed --scaling z_score --ensemble p1_p2_diff

#### Deep Learning
python3 a5task3.py -s deep_learning --main_folder ../../../Assignment_3/task2/input_pairs --n 80 --m 40 --include_features

#### Packet Analysis
python3 a5task3.py -s packet_analysis --base_directory ../../../Assignment_2/task1/pcap --output_file ../../../Assignment_2/task1/output/task_1a_analysis_results.txt

#### Statistical Analysis
python3 a5task3.py -s statistical_analysis --base_directory ../../../Assignment_2/task1 --scenario 1

#### Synthetic Data Generation
python3 a5task3.py -s synthetic_data --data_file ../../../Assignment_2/task2/dataset/csv/packet_size/scenario1_packet_size.csv --synthetic_percentage 20 --k_values 3 --normalization min_max --output_path ../../../Assignment_2/task2/output/

#### Manual Feature Generation
python3 a5task3.py -s manual_feature_gen --base_directory ../../../Assignment_2/task3/dataset/packet_size_and_direction --file_index 1 --m 10

#### Flow to CSV
python3 a5task3.py -s flow_to_csv --base_directory ../../../Assignment_1 --scenario 1

#### Calculate Euclidean Distance
python3 a5task3.py -s calculate_euclidean --scenario scenario_6 --base_path ../../../Assignment_1/output

#### Generate Histogram
python3 a5task3.py -s generate_histogram --base_path ../../../Assignment_1/output

#### Plot Traditional ML Closed World Accuracy
python3 a5task3.py -s plot_traditional_ml_closed_world_accuracy --folder ../../../Assignment_4/task1/dataset/packet_size_direction --k 10 --scenario closed --scaling min_max --ensemble p1_p2_diff --n 45

#### Plot Traditional ML Closed World Synthetic Accuracy
python3 a5task3.py -s plot_traditional_ml_closed_world_synthetic_accuracy --folder ../../../Assignment_4/task1/dataset/packet_size_direction --k 10 --scenario closed --scaling min_max --ensemble p1_p2_diff --n 45

#### Plot Venn Diagram and Runtime Memory
python3 a5task3.py -s plot_venn_diagram_and_runtime_memory --input_dir ../../../Assignment_4/task1/output/ --input_file 1a_n10_data.json --output_dir ../../../Assignment_4/task2/output

#### Plot Feature Importance
python3 a5task3.py -s plot_feature_importance --folder ../../../Assignment_4/task3/dataset/packet_size_direction --k 10 --scenario closed --scaling min_max --n 45

#### Plot Deep Learning Closed World Accuracy
python3 a5task3.py -s plot_deep_learning_closed_world_accuracy --main_folder ../../task1/dataset/packet_size_direction

#### Plot Deep Learning Closed World Synthetic Accuracy
python3 a5task3.py -s plot_deep_learning_closed_world_synthetic_accuracy --main_folder ../../task1/dataset/packet_size_direction

#### Plot Traditional ML Open World Precision Recall
python3 a5task3.py -s plot_traditional_ml_open_world_precision_recall --folder ../../task2/dataset/packet_size_direction --k 5 --scaling min_max --ensemble p1_p2_diff --n 45 --m 150

#### Plot Traditional ML Open World Synthetic Precision Recall
python3 a5task3.py -s plot_traditional_ml_open_world_synthetic_precision_recall --folder ../../task2/dataset/packet_size_direction --k 5 --scaling min_max --ensemble p1_p2_diff --n 45 --m 150

#### Plot Deep Learning Open World Precision Recall
python3 a5task3.py -s plot_deep_learning_open_world_precision_recall --main_folder ../../task2/dataset/packet_size_direction

#### Plot Deep Learning Open World Synthetic Precision Recall
python3 a5task3.py -s plot_deep_learning_open_world_synthetic_precision_recall --main_folder ../../task2/dataset/packet_size_direction
