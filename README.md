# ASR project barebones

## Installation guide

Copy this repo.

```shell
pip install -r ./requirements.txt
gdown --fuzzy https://drive.google.com/file/d/1x2eQphWpaE6B1l6OxYQerhhcI3ctx3WG/view?usp=sharing -O final_data/model.pth
```

Download  ASVSpoof 2019 Dataset from Kaggle and place LA folder in `data/dataset/LA` there we have LA files.

You can use my Google Disk link.

```
gdown --fuzzy https://drive.google.com/file/d/1-itpc_8Eihd7SKJCPkOR4xbtjKzGWF4X/view?usp=sharing
unzip data.zip
```

Then run scripts using `train.py` and `test.py`. `test.py` will print probabilities for 
each file in CMD. script `count_eer.py` counts EER on eval dataset with the same model.

Learning configs are located in `learning_configs`, and final learning script is `learning_configs/config_no_abs.json`



## Wandb Report

[Link to report](https://wandb.ai/svak/AntiSpoofing/reports/RawNet2--Vmlldzo2Mjc5MzY5?accessToken=wipdbqb0c6n1k64e4qg4agw3cgad2x5045ebha1xp48tdqa2eyoprm2e1nas9fv2)

