## 本地运行
## Build Market
```
# 更改用户名
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/zhaozc/Learnware-Private python build_market.py
```
## workflow
```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/zhaozc/Learnware-Private python workflow.py llm_example medical
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/zhaozc/Learnware-Private python workflow.py llm_example math
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/zhaozc/Learnware-Private python workflow.py llm_example finance
```
## workflow not skip eval
```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/zhaozc/Learnware-Private python workflow.py llm_example medical --rebuild False --skip_eval False
```