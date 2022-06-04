import h5py
from torch.utils import data


class MyDataset(data.Dataset):

    def __init__(self, archive, transform=None):
        self.archive = h5py.File(archive, 'r')
        self.archive = archive
        f = h5py.File(archive, 'r')
        self.f = f
        self.transform = transform

    def __getitem__(self, index):
        # datum = self.data[index]
        return self.f
        

    def __len__(self):
        return len(self.archive)

    def close(self):
        self.archive.close()

    def get_file(self):
        return self.f

