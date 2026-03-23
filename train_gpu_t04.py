import os
import argparse

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, required=True)
    args = parser.parse_args()

    env_path = args.env_path
    print("======")
    os.system(f"sh {env_path}")
    print("Endendendendend!!!!!!")