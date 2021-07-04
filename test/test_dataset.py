import unittest
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append('.')
from datasets.mvtec import MVTecDataset


class TestMyDataset(unittest.TestCase):
    def test_dataset(self):
        datafolder = 'data/mvtec'
        dataset = MVTecDataset(datafolder, is_train=True)
        print('length:', len(dataset))
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        for (x, y, mask) in tqdm(dataloader):
            # pass
            print('x shape:', x.shape)
            print('y shape:', y.shape)
            print('mask shape:', mask.shape)


if __name__ == "__main__":
    unittest.main()
