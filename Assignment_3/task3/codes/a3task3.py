import argparse
import os
import sys
from importlib import import_module

assignment3_task1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../task1/codes"))
assignment2_task2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Assignment_2/task2/codes"))
assignment1_task_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Assignment_1/codes"))

sys.path.insert(0, assignment3_task1_path)
sys.path.insert(0, assignment2_task2_path)
sys.path.insert(0, assignment1_task_path)


assignment3_task1 = import_module("a3task1")
assignment3_task1_main = assignment3_task1.main

assignment2_task2 = import_module("a2task2")
assignment2_task2_main = assignment2_task2.main

assignment1_task3d = import_module("a1task3d")
assignment1_task3d_main = assignment1_task3d.main

assignment1_task3e = import_module("a1task3e")
assignment1_task3e_main = assignment1_task3e.main

def print_overview():
    print("""
Toolbox Overview:
This toolbox combines multiple functionalities for traffic analysis and classification tasks.

Available Modes (-s <keyword>):
  - split_traffic: Split network traffic into flows and analyze these flows (Task 3, Practical Sheet 1).
  - analyze_traffic: Analyze collected traffic (Task 1, Practical Sheet 2).
  - generate_dataset: Generate a synthetic dataset using statistical methods (Practical Sheet 2).
  - classify: Perform classification using machine learning models (Previous Tasks).

    """)



def main():
    parser = argparse.ArgumentParser(
        description="Toolbox for Traffic Analysis and Classification",
        add_help=False
    )
    parser.add_argument(
        "-s",
        "--start",
        choices=["traditional_ml", "cnn", "synthetic_data", "calculate_euclidean", "generate_histogram"],
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
    elif args.start == "cnn":
        print("test")
        # split_network_traffic(remaining_args)
    elif args.start == "synthetic_data":
        sys.argv = ["a2task2.py"] + remaining_args  # Mimic calling task1.py
        assignment2_task2_main()
    elif args.start == "calculate_euclidean":
        sys.argv = ["a1task3d.py"] + remaining_args  # Mimic calling task1.py
        assignment1_task3d_main()
    elif args.start == "generate_histogram":
        sys.argv = ["a1task3e.py"] + remaining_args  # Mimic calling task1.py
        assignment1_task3e_main()
    else:
        print("Invalid mode selected. Use -h for help.")
        sys.exit(1)


if __name__ == "__main__":
    main()
