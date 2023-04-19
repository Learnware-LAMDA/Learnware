import os
import json
import sqlite3
from copy import deepcopy

from ..logger import get_module_logger
from ..learnware import get_learnware_from_dirpath
from ..config import C

logger = get_module_logger("database_ops")


def init_empty_db(func):
    def wrapper(market_id, *args, **kwargs):
        conn = sqlite3.connect(os.path.join(C.database_path, f"market_{market_id}.db"))
        cur = conn.cursor()
        listOfTables = cur.execute(
            """SELECT name FROM sqlite_master WHERE type='table' AND name='LEARNWARE'; """
        ).fetchall()
        if len(listOfTables) == 0:
            logger.info("Initializing Database in %s..." % (os.path.join(C.database_path, f"market_{market_id}.db")))
            cur.execute(
                """CREATE TABLE LEARNWARE
            (ID CHAR(10) PRIMARY KEY     NOT NULL,
            SEMANTIC_SPEC            TEXT     NOT NULL,
            ZIP_PATH     TEXT NOT NULL,
            FOLDER_PATH         TEXT NOT NULL,
            USE_FLAG         TEXT NOT NULL);"""
            )
            logger.info("Database Built!")
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
    logger.warning("!!! Drop Learnware Table !!!")
    cur.execute("DROP TABLE LEARNWARE")


@init_empty_db
def add_learnware_to_db(id: str, semantic_spec: dict, zip_path: str, folder_path: str, use_flag: str, cur):
    semantic_spec_str = json.dumps(semantic_spec)
    cur.execute(
        "INSERT INTO LEARNWARE (ID,SEMANTIC_SPEC,ZIP_PATH,FOLDER_PATH,USE_FLAG) \
      VALUES ('%s', '%s', '%s', '%s', '%s')"
        % (id, semantic_spec_str, zip_path, folder_path, use_flag)
    )


@init_empty_db
def delete_learnware_from_db(id: str, cur):
    cur.execute("DELETE from LEARNWARE where ID='%s';" % (id))


@init_empty_db
def load_market_from_db(cur):
    logger.info("Reload from Database")
    cursor = cur.execute("SELECT id, semantic_spec, zip_path, FOLDER_PATH from LEARNWARE")

    learnware_list = {}
    zip_list = {}
    folder_list = {}
    max_count = 0

    for id, semantic_spec, zip_path, folder_path in cursor:
        semantic_spec_dict = json.loads(semantic_spec)
        new_learnware = get_learnware_from_dirpath(
            id=id, semantic_spec=semantic_spec_dict, learnware_dirpath=folder_path
        )

        learnware_list[id] = new_learnware
        zip_list[id] = zip_path
        folder_list[id] = folder_path
        max_count = max(max_count, int(id))

    logger.info("Market Reloaded from DB.")
    return learnware_list, zip_list, folder_list, max_count + 1
