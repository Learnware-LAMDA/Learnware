import sqlite3
import os
from ..logger import get_module_logger
from ..learnware import Learnware
from ..specification import RKMEStatSpecification, Specification
import json

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT_PATH, "market.db")
LOGGER = get_module_logger("db")


def init_empty_db(func):
    def wrapper(*args, **kwargs):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        listOfTables = cur.execute(
            """SELECT name FROM sqlite_master WHERE type='table' AND name='LEARNWARE'; """
        ).fetchall()
        if listOfTables == []:
            LOGGER.info("Initializing Database in %s..." % (DB_PATH))
            cur.execute(
                """CREATE TABLE LEARNWARE
            (ID CHAR(10) PRIMARY KEY     NOT NULL,
            NAME           TEXT    NOT NULL,
            SEMANTIC_SPEC            TEXT     NOT NULL,
            MODEL_PATH     TEXT NOT NULL,
            STAT_SPEC_PATH         TEXT NOT NULL);"""
            )
            LOGGER.info("Database Built!")
        kwargs["cur"] = cur
        item = func(*args, **kwargs)
        conn.commit()
        conn.close()
        return item

    return wrapper


# Clear Learnware Database
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!                                    !!!!!
# !!!!! Do NOT use unless highly necessary !!!!!
# !!!!!                                    !!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@init_empty_db
def clear_learnware_table(cur):
    LOGGER.warning("!!! Drop Learnware Table !!!")
    cur.execute("DROP TABLE LEARNWARE")


@init_empty_db
def add_learnware_to_db(id: str, name: str, model_path: str, stat_spec_path: str, semantic_spec: dict, cur):
    semantic_spec_str = json.dumps(semantic_spec)
    stat_spec_path_dict = {"RKME": stat_spec_path}
    stat_spec_str = json.dumps(stat_spec_path_dict)
    cur.execute(
        "INSERT INTO LEARNWARE (ID,NAME,SEMANTIC_SPEC,MODEL_PATH,STAT_SPEC_PATH) \
      VALUES ('%s', '%s', '%s', '%s', '%s' )"
        % (id, name, semantic_spec_str, model_path, stat_spec_str)
    )


@init_empty_db
def delete_learnware_from_db(id: str, cur):
    cur.execute("DELETE from LEARNWARE where ID='%s';" % (id))


@init_empty_db
def load_market_from_db(cur):
    LOGGER.info("Reload from Database")
    cursor = cur.execute("SELECT id, name, semantic_spec, model_path, stat_spec_path from LEARNWARE")

    learnware_list = {}
    max_count = 0
    for item in cursor:
        id, name, semantic_spec, model_path, stat_spec_path = item
        semantic_spec_dict = json.loads(semantic_spec)
        stat_spec_path_dict = json.loads(stat_spec_path)
        stat_spec_dict = {}
        for stat_spec_name in stat_spec_path_dict:
            new_stat_spec = RKMEStatSpecification()
            new_stat_spec.load(stat_spec_path_dict[stat_spec_name])
            stat_spec_dict[stat_spec_name] = new_stat_spec
        # Commented for test purpose. Uncomment when Learnware class is implemented.
        # model_dict = {"module_path": model_path, "class_name": "BaseModel"}
        model_dict = model_path
        specification = Specification(semantic_spec=semantic_spec_dict, stat_spec=stat_spec_dict)
        new_learnware = Learnware(id=id, name=name, model=model_dict, specification=specification)
        learnware_list[id] = new_learnware
        max_count = max(max_count, int(id))
    LOGGER.info("Market Reloaded from DB.")
    return learnware_list, max_count + 1
