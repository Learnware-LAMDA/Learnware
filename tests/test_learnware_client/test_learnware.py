from learnware.client.learnware_client import LearnwareClient
from learnware.test import get_semantic_specification

if __name__ == "__main__":
    semantic_specification = get_semantic_specification()

    zip_path = "test.zip"
    client = LearnwareClient()
    client.install_environment(zip_path)
    client.test_learnware(zip_path, semantic_specification)
