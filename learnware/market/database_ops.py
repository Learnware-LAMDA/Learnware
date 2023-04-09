import os
import json
import sqlite3

from ..logger import get_module_logger
from ..learnware import get_learnware_from_dirpath


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
            SEMANTIC_SPEC            TEXT     NOT NULL,
            ZIP_PATH     TEXT NOT NULL,
            FOLDER_PATH         TEXT NOT NULL);"""
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
def add_learnware_to_db(id: str, semantic_spec: dict, zip_path: str, folder_path: str, cur):
    semantic_spec_str = json.dumps(semantic_spec)
    cur.execute(
        "INSERT INTO LEARNWARE (ID,SEMANTIC_SPEC,ZIP_PATH,FOLDER_PATH) \
      VALUES ('%s', '%s', '%s', '%s' )"
        % (id, semantic_spec_str, zip_path, folder_path)
    )


@init_empty_db
def delete_learnware_from_db(id: str, cur):
    cur.execute("DELETE from LEARNWARE where ID='%s';" % (id))


@init_empty_db
def load_market_from_db(cur):
    LOGGER.info("Reload from Database")
    cursor = cur.execute("SELECT id, semantic_spec, zip_path, FOLDER_PATH from LEARNWARE")

    learnware_list = {}
    zip_list = {}
    folder_list = {}
    max_count = 0
    for item in cursor:
        id, semantic_spec, zip_path, folder_path = item
        semantic_spec_dict = json.loads(semantic_spec)
        new_learnware = get_learnware_from_dirpath(
            id=id, semantic_spec=semantic_spec_dict, learnware_dirpath=folder_path
        )
        learnware_list[id] = new_learnware
        zip_list[id] = zip_path
        folder_list = folder_path
        max_count = max(max_count, int(id))
    LOGGER.info("Market Reloaded from DB.")
    return learnware_list, zip_list, folder_list, max_count + 1
