import hashlib
import requests
import os
import random
import json
import time
from tqdm import tqdm

email = "tanzh@lamda.nju.edu.cn"
password = hashlib.md5(b"Qwerty123").hexdigest()
login_url = "http://210.28.134.201:8089/auth/login"
submit_url = "http://210.28.134.201:8089/user/add_learnware"
all_data_type = ["Table", "Image", "Video", "Text", "Audio"]
all_task_type = [
    "Classification",
    "Regression",
    "Clustering",
    "Feature Extraction",
    "Generation",
    "Segmentation",
    "Object Detection",
]
all_device_type = ["CPU", "GPU"]
all_scenario = [
    "Business",
    "Financial",
    "Health",
    "Politics",
    "Computer",
    "Internet",
    "Traffic",
    "Nature",
    "Fashion",
    "Industry",
    "Agriculture",
    "Education",
    "Entertainment",
    "Architecture",
]

# ###############
# 以上部分无需修改 #
# ###############


def main():
    session = requests.Session()
    res = session.post(login_url, json={"email": email, "password": password})

    # /path/to/learnware/folder 修改为学件文件夹地址
    learnware_pool = os.listdir(os.path.join(os.path.abspath("."), "learnware_pool"))

    for learnware in learnware_pool:
        # 修改相应的语义规约
        name = "M5_Shop" + "%02d" % int(learnware.split(".")[0].split("_")[1])
        name = name + "_" + time.strftime("%Y%m%d%H%M%S", time.localtime())
        description = f"This is a description of learnware {name}"
        data = random.choice(all_data_type)
        task = random.choice(all_task_type)
        device = list(set(random.choices(all_device_type, k=2)))
        scenario = list(set(random.choices(all_scenario, k=5)))
        semantic_specification = {
            "Data": {"Values": ["Table"], "Type": "Class"},
            "Task": {"Values": ["Regression"], "Type": "Class"},
            "Device": {"Values": ["CPU"], "Type": "Tag"},
            "Scenario": {"Values": ["Business"], "Type": "Tag"},
            "Description": {"Values": "A sales-forecasting model from Walmart store", "Type": "String"},
            "Name": {"Values": name, "Type": "String"},
            "License": {"Values": ["MIT"], "Type": "Class"},
        }
        res = session.post(
            submit_url,
            data={
                "semantic_specification": json.dumps(semantic_specification),
            },
            files={
                "learnware_file": open(
                    os.path.join(os.path.abspath("."), "learnware_pool", learnware),
                    "rb",
                )
            },
        )
        assert json.loads(res.text)["code"] == 0, "Upload error"


if __name__ == "__main__":
    main()
