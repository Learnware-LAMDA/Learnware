import fire
from pyinstrument import Profiler

from dataset_corporacion_workflow import CorporacionDatasetWorkflow
from dataset_heterogeneous_workflow import HeterogeneousWorkflow

workflow_mapping = {
    "test_homo_unlabeled": CorporacionDatasetWorkflow,
    "test_homo_labeled": CorporacionDatasetWorkflow,
    "test_hetero_unlabeled": HeterogeneousWorkflow,
    "test_hetero_labeled": HeterogeneousWorkflow,
}


def main():
    def dispatch(command):
        if command in workflow_mapping:
            workflow_class = workflow_mapping[command]
            workflow_instance = workflow_class()
            getattr(workflow_instance, command)()
        else:
            print(f"No workflow found for command: {command}")

    fire.Fire(dispatch)


if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()

    main()

    profiler.stop()
    profiler.print()
