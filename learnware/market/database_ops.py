import sqlite3
import os
from ..logger import get_module_logger
from ..learnware import Learnware
from ..specification import RKMEStatSpecification, Specification
import json

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT_PATH, "market.db")
LOGGER = get_module_logger("market")


def init_empty_db(func):
    def wrapper():
        if not os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            LOGGER.info("Initializing Database in %s..." % (DB_PATH))
            c = conn.cursor()
            c.execute(
                """CREATE TABLE LEARNWARE
            (ID CHAR(10) PRIMARY KEY     NOT NULL,
            NAME           TEXT    NOT NULL,
            SEMANTIC_SPEC            TEXT     NOT NULL,
            MODEL_PATH     TEXT NOT NULL,
            STAT_SPEC_PATH         TEXT NOT NULL);"""
            )
            LOGGER.info("Database Built!")
            conn.commit()
            conn.close()
        func()

    return wrapper


@init_empty_db
def add_learnware_to_db():
    pass


@init_empty_db
def delete_learnware_from_db(id: str):
    pass


@init_empty_db
def load_market_from_db():
    conn = sqlite3.connect(DB_PATH)
    LOGGER.info("Reload from Database")
    c = conn.cursor()
    cursor = c.execute("SELECT id, name, semantic_spec, model_path, stat_spec_path from LEARNWARE")

    learnware_list = {}
    max_count = 0
    for item in cursor:
        id, name, semantic_spec, model_path, stat_spec_path = item
        semantic_spec_dict = json.loads(semantic_spec)
        stat_spec_path_dict = json.loads(stat_spec_path)
        stat_spec_dict = {}
        for stat_spec_name in stat_spec_path_dict:
            new_stat_spec = RKMEStatSpecification()
            new_stat_spec.load(stat_spec_dict[stat_spec_name])
            stat_spec_dict[stat_spec_name] = new_stat_spec
        model_dict = {"model_path": model_path, "class_name": "BaseModel"}
        specification = Specification(semantic_spec=semantic_spec_dict, stat_spec=stat_spec_dict)
        new_learnware = Learnware(id=id, name=name, model=model_dict, specification=specification)
        learnware_list[id] = new_learnware
        max_count = max(max_count, int(id))
    conn.commit()
    conn.close()
    LOGGER.info("Market Reloaded from DB.")
    return learnware_list, max_count
