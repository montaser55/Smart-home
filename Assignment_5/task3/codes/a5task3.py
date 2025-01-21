import argparse
import os
import sys
from importlib import import_module

assignment5_task1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../task1/codes"))
assignment5_task2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../task2/codes"))
assignment4_task1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Assignment_4/task1/codes"))
assignment4_task2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Assignment_4/task2/codes"))
assignment4_task3_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Assignment_4/task3/codes"))
assignment3_task1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Assignment_3/task1/codes"))
assignment3_task2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Assignment_3/task2/codes"))
assignment2_task1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Assignment_2/task1/codes"))
assignment2_task2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Assignment_2/task2/codes"))
assignment2_task3_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Assignment_2/task3/codes"))
assignment1_task_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Assignment_1/codes"))

sys.path.insert(0, assignment5_task1_path)
sys.path.insert(0, assignment5_task2_path)
sys.path.insert(0, assignment4_task1_path)
sys.path.insert(0, assignment4_task2_path)
sys.path.insert(0, assignment4_task3_path)
sys.path.insert(0, assignment3_task1_path)
sys.path.insert(0, assignment3_task2_path)
sys.path.insert(0, assignment2_task1_path)
sys.path.insert(0, assignment2_task2_path)
sys.path.insert(0, assignment2_task3_path)
sys.path.insert(0, assignment1_task_path)

assignment5_task2d = import_module("a5task2d")
assignment5_task2d_main = assignment5_task2d.main

assignment5_task2c = import_module("a5task2c")
assignment5_task2c_main = assignment5_task2c.main

assignment5_task2b = import_module("a5task2b")
assignment5_task2b_main = assignment5_task2b.main

assignment5_task2a = import_module("a5task2a")
assignment5_task2a_main = assignment5_task2a.main

assignment5_task1c = import_module("a5task1c")
assignment5_task1c_main = assignment5_task1c.main

assignment5_task1ab = import_module("a5task1ab")
assignment5_task1ab_main = assignment5_task1ab.main

assignment4_task3 = import_module("a4task3")
assignment4_task3_main = assignment4_task3.main

assignment4_task2 = import_module("a4task2")
assignment4_task2_main = assignment4_task2.main

assignment4_task1cd = import_module("a4task1cd")
assignment4_task1cd_main = assignment4_task1cd.main

assignment4_task1ab = import_module("a4task1ab")
assignment4_task1ab_main = assignment4_task1ab.main

assignment3_task1 = import_module("a3task1")
assignment3_task1_main = assignment3_task1.main

assignment3_task2 = import_module("a3task2")
assignment3_task2_main = assignment3_task2.main

assignment2_task1a = import_module("a2task1a")
assignment2_task1a_main = assignment2_task1a.main

assignment2_task1b1c = import_module("a2task1b1c")
assignment2_task1b1c_main = assignment2_task1b1c.main

assignment2_task2 = import_module("a2task2")
assignment2_task2_main = assignment2_task2.main

assignment2_task3 = import_module("a2task3")
assignment2_task3_main = assignment2_task3.main

assignment1_task3c = import_module("a1task3c")
assignment1_task3c_main = assignment1_task3c.main

assignment1_task3d = import_module("a1task3d")
assignment1_task3d_main = assignment1_task3d.main

assignment1_task3e = import_module("a1task3e")
assignment1_task3e_main = assignment1_task3e.main


def print_overview():
    print("""
Toolbox Overview:
This toolbox combines multiple functionalities for traffic analysis and classification tasks.

Available Modes (-s <keyword>):
  - traditional_ml: Use traditional machine learning for device classification.
  - deep_learning: Use deep learning for device classification.
  - packet_analysis: Analyzes files to count protocols and message types, calculates their fractions.
  - statistical_analysis: Analyzes files to compute and visualize packet length statistics, inter-arrival times, CDFs, and CCDFs.
  - synthetic_data: Generate Synthetic Data.
  - manual_feature_gen: Processes packets and plots features.
  - flow_to_csv: Make flows into CSVs.
  - calculate_euclidean: Calculate euclidean distance among files.
  - generate_histogram: Generate Histogram of data.
  - plot_traditional_ml_closed_world_accuracy: Plot accuracy results for traditional machine learning models in a closed-world scenario.
  - plot_traditional_ml_closed_world_synthetic_accuracy: Plot accuracy results for traditional machine learning models in a closed-world scenario using synthetic data.
  - plot_venn_diagram_and_runtime_memory: Create Venn diagrams to visualize overlaps between datasets and analyze runtime memory usage.
  - plot_feature_importance: Plot the importance of features used in traditional machine learning models.
  - plot_deep_learning_closed_world_accuracy: Plot accuracy results for deep learning models in a closed-world scenario.
  - plot_deep_learning_closed_world_synthetic_accuracy: Plot accuracy results for deep learning models in a closed-world scenario using synthetic data.
  - plot_traditional_ml_open_world_precision_recall: Plot precision and recall results for traditional machine learning models in an open-world scenario.
  - plot_traditional_ml_open_world_synthetic_precision_recall: Plot precision and recall results for traditional machine learning models in an open-world scenario using synthetic data.
  - plot_deep_learning_open_world_precision_recall: Plot precision and recall results for deep learning models in an open-world scenario.
  - plot_deep_learning_open_world_synthetic_precision_recall: Plot precision and recall results for deep learning models in an open-world scenario using synthetic data.

    """)



def main():
    parser = argparse.ArgumentParser(
        description="Toolbox for all the tasks",
        add_help=False
    )
    parser.add_argument(
        "-s",
        "--start",
        choices=["traditional_ml", "deep_learning", "packet_analysis", "statistical_analysis", "synthetic_data",
                 "manual_feature_gen",
                 "flow_to_csv", "calculate_euclidean", "generate_histogram",
                 "plot_traditional_ml_closed_world_accuracy", "plot_traditional_ml_closed_world_synthetic_accuracy",
                 "plot_venn_diagram_and_runtime_memory", "plot_feature_importance", "plot_deep_learning_closed_world_accuracy",
                 "plot_deep_learning_closed_world_synthetic_accuracy", "plot_traditional_ml_open_world_precision_recall",
                 "plot_traditional_ml_open_world_synthetic_precision_recall", "plot_deep_learning_open_world_precision_recall",
                 "plot_deep_learning_open_world_synthetic_precision_recall"],
        help="Select the operation mode for the toolbox."
    )
    parser.add_argument(
        "-h",
        "--help_mode",
        action="store_true",
        help="Show help information for the selected mode."
    )

    args, remaining_args = parser.parse_known_args()

    if args.help_mode:
        print_overview()
        sys.exit(0)

    if args.start == "traditional_ml":
        sys.argv = ["a3task1.py"] + remaining_args
        assignment3_task1_main()
    elif args.start == "deep_learning":
        sys.argv = ["a3task2.py"] + remaining_args
        assignment3_task2_main()
    elif args.start == "packet_analysis":
        sys.argv = ["a2task1a.py"] + remaining_args
        assignment2_task1a_main()
    elif args.start == "statistical_analysis":
        sys.argv = ["a2task1b1c.py"] + remaining_args
        assignment2_task1b1c_main()
    elif args.start == "synthetic_data":
        sys.argv = ["a2task2.py"] + remaining_args
        assignment2_task2_main()
    elif args.start == "manual_feature_gen":
        sys.argv = ["a2task3.py"] + remaining_args
        assignment2_task3_main()
    elif args.start == "flow_to_csv":
        sys.argv = ["a1task3c.py"] + remaining_args
        assignment1_task3c_main()
    elif args.start == "calculate_euclidean":
        sys.argv = ["a1task3d.py"] + remaining_args
        assignment1_task3d_main()
    elif args.start == "generate_histogram":
        sys.argv = ["a1task3e.py"] + remaining_args
        assignment1_task3e_main()
    elif args.start == "plot_traditional_ml_closed_world_accuracy":
        sys.argv = ["a4task1ab.py"] + remaining_args
        assignment4_task1ab_main()
    elif args.start == "plot_traditional_ml_closed_world_synthetic_accuracy":
        sys.argv = ["a4task1cd.py"] + remaining_args
        assignment4_task1cd_main()
    elif args.start == "plot_venn_diagram_and_runtime_memory":
        sys.argv = ["a4task2.py"] + remaining_args
        assignment4_task2_main()
    elif args.start == "plot_feature_importance":
        sys.argv = ["a4task3.py"] + remaining_args
        assignment4_task3_main()
    elif args.start == "plot_deep_learning_closed_world_accuracy":
        sys.argv = ["a5task1ab.py"] + remaining_args
        assignment5_task1ab_main()
    elif args.start == "plot_deep_learning_closed_world_synthetic_accuracy":
        sys.argv = ["a5task1c.py"] + remaining_args
        assignment5_task1c_main()
    elif args.start == "plot_traditional_ml_open_world_precision_recall":
        sys.argv = ["a5task2a.py"] + remaining_args
        assignment5_task2a_main()
    elif args.start == "plot_traditional_ml_open_world_synthetic_precision_recall":
        sys.argv = ["a5task2b.py"] + remaining_args
        assignment5_task2b_main()
    elif args.start == "plot_deep_learning_open_world_precision_recall":
        sys.argv = ["a5task2c.py"] + remaining_args
        assignment5_task2c_main()
    elif args.start == "plot_deep_learning_open_world_synthetic_precision_recall":
        sys.argv = ["a5task2d.py"] + remaining_args
        assignment5_task2d_main()
    else:
        print("Invalid mode selected. Use -h for help.")
        sys.exit(1)


if __name__ == "__main__":
    main()
