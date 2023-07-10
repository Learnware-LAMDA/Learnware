from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, text
from sqlalchemy import (
    Column, Integer, Text, DateTime, String
)
import os
import json
from ..learnware import get_learnware_from_dirpath


DeclarativeBase = declarative_base()


class Learnware(DeclarativeBase):
    __tablename__ = 'tb_learnware'

    id = Column(String(10), primary_key=True, nullable=False)
    semantic_spec = Column(Text, nullable=False)
    zip_path = Column(Text, nullable=False)
    folder_path = Column(Text, nullable=False)
    use_flag = Column(Text, nullable=False)

    pass


class DatabaseOperations(object):

    def __init__(self, url: str, database_name: str):
        if url.startswith("sqlite"):
            url = os.path.join(url, f"{database_name}.db")
        else:
            url = f"{url}/{database_name}"
            pass

        self.url = url
        self.create_database_if_not_exists(url)

        pass

    
    def create_database_if_not_exists(self, url):
        database_exists = True

        if url.startswith("sqlite"):
            # it is sqlite
            start = url.find(":///")
            path = url[start+4:]
            if os.path.exists(path):
                database_exists = True
                pass
            else:
                database_exists = False
                os.makedirs(os.path.dirname(path), exist_ok=True)
                pass
            pass
        elif self.url.startswith("postgresql"):
            # it is postgresql
            dbname_start = url.rfind("/")
            dbname = url[dbname_start+1:]
            url_no_dbname = url[:dbname_start]
            engine = create_engine(url_no_dbname)

            with engine.connect() as conn:
                result = conn.execute(text("SELECT datname FROM pg_database;"))
                db_list = set()

                for row in  result.fetchall():
                    db_list.add(row[0].lower())
                    pass

                if dbname.lower() not in db_list:
                    database_exists = False
                    conn.execution_options(isolation_level="AUTOCOMMIT").execute(
                    text("CREATE DATABASE {0};".format(dbname)))
                    pass
                else:
                    database_exists = True
                    pass
                pass
            engine.dispose()
            pass
        else:
            raise Exception(f"Unsupported database url: {self.url}")
            pass
        
        self.engine = create_engine(url, future=True)

        if not database_exists:
            DeclarativeBase.metadata.create_all(self.engine)
            pass
        pass

    def clear_learnware_table(self):
        with self.engine.connect() as conn:
            conn.execute(text("DELETE FROM tb_learnware;"))
            conn.commit()
            pass
        pass

    def add_learnware(self, id: str, semantic_spec: dict, zip_path, folder_path, use_flag: str):
        with self.engine.connect() as conn:
            semantic_spec_str = json.dumps(semantic_spec)
            conn.execute(
                text(
                ("INSERT INTO tb_learnware (id, semantic_spec, zip_path, folder_path, use_flag)"
                 "VALUES (:id, :semantic_spec, :zip_path, :folder_path, :use_flag);")
                ),
                dict(id=id, semantic_spec=semantic_spec_str, zip_path=zip_path,
                folder_path=folder_path, use_flag=use_flag)
            )
            conn.commit()
            pass
        pass
    
    def delete_learnware(self, id: str):
        with self.engine.connect() as conn:
            conn.execute(
                text("DELETE FROM tb_learnware WHERE id=:id;"),
                dict(id=id)
            )
            conn.commit()
            pass
        pass

    def load_market(self):
        with self.engine.connect() as conn:
            cursor = conn.execute(text("SELECT id, semantic_spec, zip_path, folder_path, use_flag FROM tb_learnware;"))

            learnware_list = {}
            zip_list = {}
            folder_list = {}
            max_count = 0

            for id, semantic_spec, zip_path, folder_path, use_flag in cursor:
                id = id.strip()
                semantic_spec_dict = json.loads(semantic_spec)
                new_learnware = get_learnware_from_dirpath(
                    id=id, semantic_spec=semantic_spec_dict, learnware_dirpath=folder_path
                )
                print(f'load learnware: {id}')
                learnware_list[id] = new_learnware
                # assert new_learnware is not None
                zip_list[id] = zip_path
                folder_list[id] = folder_path
                max_count = max(max_count, int(id))
            pass

        return learnware_list, zip_list, folder_list, max_count + 1
        pass

    pass