from torch.utils.data import Dataset

class testDataset(Dataset):
    def __init__(self):
        print('initializing')

    def __getitem__(self, item):
        pass
    def __len__(self):
        pass

testDataset()
