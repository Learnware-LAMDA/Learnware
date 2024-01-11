import fire

from learnware.logger import get_module_logger
from homo import HomogeneousDatasetWorkflow
from hetero import HeterogeneousDatasetWorkflow
from config import homo_table_benchmark_config, hetero_cross_feat_eng_benchmark_config, hetero_cross_task_benchmark_config

logger = get_module_logger("base_table", level="INFO")


class TableDatasetWorkflow:
    def unlabeled_homo_table_example(self):
        workflow = HomogeneousDatasetWorkflow(
            benchmark_config=homo_table_benchmark_config,
            name="easy",
            rebuild=True
        )
        workflow.unlabeled_homo_table_example()

    def labeled_homo_table_example(self):
        workflow = HomogeneousDatasetWorkflow(
            benchmark_config=homo_table_benchmark_config,
            name="easy",
            rebuild=False
        )
        workflow.labeled_homo_table_example()
    
    def cross_feat_eng_hetero_table_example(self):
        workflow = HeterogeneousDatasetWorkflow(
            benchmark_config=hetero_cross_feat_eng_benchmark_config,
            name="hetero",
            rebuild=True
        )
        workflow.unlabeled_hetero_table_example()

    def cross_task_hetero_table_example(self):
        workflow = HeterogeneousDatasetWorkflow(
            benchmark_config=hetero_cross_task_benchmark_config,
            name="hetero",
            rebuild=False
        )
        workflow.labeled_hetero_table_example()


if __name__ == "__main__":
    fire.Fire(TableDatasetWorkflow)