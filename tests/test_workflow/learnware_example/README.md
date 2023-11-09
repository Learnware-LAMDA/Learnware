## How to Generate Environment Yaml

* create env config for conda:
```shell
conda env export | grep -v "^prefix: " > environment.yml
```
* recover env from config
```
conda env create -f environment.yml
```