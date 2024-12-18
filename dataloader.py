from torch.utils.data import Dataset, DataLoader
import torch
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_lenght, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids)- max_lenght, stride):
            input_chunk = token_ids[i:i + max_lenght]
            target_chunk = token_ids[i + 1: i + max_lenght + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_lenght=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_lenght, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,    # Drop last batch if it is shorter than the specified batch_size to prevent loss spikes during training
        num_workers=num_workers
    )

    return dataloader