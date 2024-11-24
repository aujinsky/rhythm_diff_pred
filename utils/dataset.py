if __name__=="__main__":
    import datareader
else:
    from utils import datareader

import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def get_random_seq(seq, seq_len):
    start = random.randrange(0, len(seq) + 1 - seq_len)
    end = start + seq_len
    return seq[start : end]


def load_data(
    dataset_type, 
    train_trans_path = None, 
    valid_trans_path = None, 
    test_trans_path = None, 
    seq_len = -1, 
    min_seq_len = -1,
    normalize = False
):
    assert dataset_type in ['osumania'], 'invalid dataset type.'

    if dataset_type == 'osumania':
        trainset = OsuManiaDataset(
            fpath = train_trans_path,
            seq_len = seq_len,
            min_seq_len = min_seq_len,
            normalize = normalize
        )
        validset = OsuManiaDataset(
            fpath = valid_trans_path,
            seq_len = seq_len,
            min_seq_len = min_seq_len,
            normalize = normalize
        )
        testset = OsuManiaDataset(
            fpath = test_trans_path,
            seq_len = seq_len,
            min_seq_len = min_seq_len,
            normalize = normalize
        )
    else:
        raise NotImplementedError
    
    return trainset, validset, testset

class OsuManiaDataset(Dataset):
    def __init__(self, fpath = 'osumania_data/mapping_list.csv', beatmap_info_path = 'osumania_data/beatmap_data.csv', n_key_only = True):
        super().__init__()

        self.path_len = len(fpath)
        self.n_key_only = n_key_only
        self.beatmap_info_path = beatmap_info_path
        self._load_data(fpath)
        self.SOME_BPM_RELATED_CONST = 60 # to be defined (250bpm 16beat)
        if self.n_key_only == True:
            self.keys = 4
        else:
            self.keys = 8 # 
        self.device = 0
        

    def _load_data(self, mapping_list_path):
        self.data_path = []
        mapping_list = pd.read_csv(mapping_list_path, header =None)
        self.sr_pd = pd.read_csv(self.beatmap_info_path, header =None).iloc[:, -1]
        self.num_of_key = pd.read_csv(self.beatmap_info_path, header =None).iloc[:, 7]
        self.data_path = list(mapping_list.iloc[:,3])
        


    def __getitem__(self, index):
        filename = self.data_path[index]
        temp = datareader.DataReader(filename)
        return {
            'id': index,
            'pattern': temp.notes,
            'len': temp.notes[-1]['end_t'], # incorrect but move on.
            'num_of_key': self.num_of_key[index],
            'sr': self.sr_pd[index]
        }

    def __len__(self):
        return len(self.data_path)

    def collate_fn(self, batch):
        max_len = -1
        length_list = []

        for item in batch:
            if self.n_key_only and self.keys == 4 and item['num_of_key'] == 4:
                if max_len < item['len']:
                    max_len = item['len']
                length_list.append(int(item['len']/ self.SOME_BPM_RELATED_CONST))
            

        pattern_tensor = torch.zeros((int(max_len / self.SOME_BPM_RELATED_CONST)+1, len(length_list), self.keys), device=self.device)
        sr_tensor = torch.zeros(len(length_list))
        count = 0
        for i, item in enumerate(batch):
            if self.n_key_only and self.keys == 4 and item['num_of_key'] == 4:
                for note in item['pattern']:
                    pattern_tensor[int(note['start_t'] / self.SOME_BPM_RELATED_CONST), count, int(note['key'])] = 1.
                sr_tensor[count] = item['sr']
                count = count + 1
        

        return pattern_tensor, [length_list, int(max_len / self.SOME_BPM_RELATED_CONST)+1], sr_tensor
        
                




if __name__ == '__main__':
    dataset_osu = OsuManiaDataset(
        fpath = 'osumania_data\mapping_list.csv'
    )
    import IPython; IPython.embed(); exit(1)
