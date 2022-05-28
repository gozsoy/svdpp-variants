from torch.utils.data import Dataset


class SVD_Dataset(Dataset):
    def __init__(self, df):
        # pd -> np
        self.df = df.values
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        item = self.df[idx]
        u_id, m_id, r = item[0], item[1], item[2]

        return (u_id, m_id), r


class SVDPP_Dataset(Dataset):
    def __init__(self, df, user_rated_items_df):
        # pd -> np
        self.df = df.values
        self.user_rated_items_df = user_rated_items_df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        item = self.df[idx]
        u_id, m_id, r = item[0], item[1], item[2]

        temp_df = self.user_rated_items_df[
            self.user_rated_items_df.user_id == u_id]
        rated_items = temp_df.movie_id.values[0]
        rated_count = temp_df.counter.values[0]

        return u_id, m_id, rated_items, rated_count, r