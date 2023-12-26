import fire

from learnware.logger import get_module_logger
from homo import CorporacionDatasetWorkflow
from config import homo_table_benchmark_config

logger = get_module_logger("base_table", level="INFO")


class TableDatasetWorkflow:
    def unlabeled_homo_table_example(self):
        workflow = CorporacionDatasetWorkflow(
            benchmark_config=homo_table_benchmark_config,
            name="easy",
            rebuild=False
        )
        workflow.unlabeled_homo_table_example()

    def labeled_homo_table_example(self):
        workflow = CorporacionDatasetWorkflow(
            benchmark_config=homo_table_benchmark_config,
            name="easy",
            rebuild=False
        )
        workflow.labeled_homo_table_example()
    
    def cross_feat_eng_hetero_table_example(self):
        pass

    def cross_task_hetero_table_example(self):
        pass


if __name__ == "__main__":
    fire.Fire(TableDatasetWorkflow)