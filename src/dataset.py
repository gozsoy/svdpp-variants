from torch.utils.data import Dataset

class CF_Dataset(Dataset):
    def __init__(self, df):
        # pd -> np
        self.df = df.values
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):

        item = self.df[idx]
        u_id, m_id, r = item[0],item[1],item[2]

        return (u_id,m_id),r