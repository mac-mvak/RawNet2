from hw_asv.datasets import ASVDataset

config_parser = {'preprocessing': {'sr':16000}}

uu = ASVDataset('train', config_parser=config_parser)
a = uu[0]
print(a['label'])
