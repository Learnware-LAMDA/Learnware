import sqlite3
import os
from ..logger import get_module_logger
from ..learnware import Learnware

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT_PATH, "market.db")
LOGGER = get_module_logger("market", level="INFO")


def add_learnware_to_db():
    pass


def delete_learnware_from_db():
    pass


def init_empty_db():
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


def load_market_from_db():
    if not os.path.exists(DB_PATH):
        init_empty_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cursor = c.execute("SELECT id, name, semantic_spec, model_path, stat_spec_path from LEARNWARE")

    for item in cursor:
        id, name, semantic_spec, model_path, stat_spec_path = item
    LOGGER.info("Market Reloaded from DB.")
