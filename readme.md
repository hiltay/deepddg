# 利用featurehub的特征计算ddg

注意，这里算的都只是单个点突变的ddg

## 安装

首先，请确保[conda](https://docs.conda.io/en/latest/miniconda.html)可以使用，并进入你期望安装所有依赖的虚拟环境下

    ```shell
    # conda install -yc conda-forge python=3.9
    conda install -yc conda-forge mamba
    mamba install -yc conda-forge -c salilab dssp pdbfixer autogluon=0.7.0
    pip install --no-cache-dir git+ssh://git@git.dp.tech/macromolecule/macrofeaturehub.git
    pip install --no-cache-dir pandarallel
    pip install --no-cache-dir -U "dp-launching-sdk"
    conda clean --all -y
    wget -r --no-parent -nH -R "index.html" --cut-dirs=1 --compression=gzip -P checkpoints --user=admin --password=LzzM5OQtTGKYHSwpQqOAn6fL7Lu1medN http://ohgy1146835.bohrium.tech:50004/ddg_model/
    pip install -e .
    ```

## 使用

    ```shell
    run_macroddg -i [包含输入信息的json] -o [输出的文件夹]
    # 例如
    # run_macroddg -i test/4b72_strange.json -o test/4b72_strange
    # 可以如下测试launching是否正常
    # python macroddg/app/launching/main.py --input_pdb test/4b72_strange.pdb --mutations "A:G499Q, A:V502N"
    ```

### 输入文件说明

样例可以见test文件夹下的各个json文件

    ```json
    {
        "pdb_path": "test/4b72_strange.pdb",
        "mutcodes": [
            "A:G499Q",
            "A:V502N",
            "A:R7V"
        ]
    }
    ```

pdb_path后输入一个待突变的原始pdb文件

mutcodes输入一个列表，列表内为字符串，命名方式为：<链>:<突变前残基><pdb中对应的id><突变后的残基>

mutcodes会逐个算ddg，而**不是**算一个多突变的ddg
