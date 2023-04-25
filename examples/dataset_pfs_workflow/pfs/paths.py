import os

ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "data"))
raw_data_dir = os.path.join(ROOT_PATH, "raw_data")
split_data_dir = os.path.join(ROOT_PATH, "split_data")
res_dir = os.path.join(ROOT_PATH, "results")
model_dir = os.path.join(ROOT_PATH, "models")
model_dir2 = os.path.join(ROOT_PATH, "models2")


for dir_name in [ROOT_PATH, raw_data_dir, split_data_dir, res_dir, model_dir, model_dir2]:
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

pfs_data_dir = os.path.join(raw_data_dir, "PFS")
pfs_split_dir = os.path.join(split_data_dir, "PFS")
pfs_res_dir = os.path.join(res_dir, "PFS")

for dir_name in [pfs_data_dir, pfs_split_dir, pfs_res_dir]:
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
