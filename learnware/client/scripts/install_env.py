import argparse

from learnware.client.utils import install_environment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learnware-dirpath", type=str, required=True, help="path of learnware dir")
    parser.add_argument("--conda-env", type=str, required=False, help="name of conda env")

    args = parser.parse_args()

    learnware_dirpath = args.learnware_dirpath
    conda_env = args.conda_env

    install_environment(learnware_dirpath, conda_env)
