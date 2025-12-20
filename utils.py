import torch
import os
from torch.utils.data import DataLoader
import random
import numpy as np
import pickle
from tqdm import tqdm


def extract_layer_mlp(model, layer_idx):
    """
    Extract the MLP layer from the specified layer index of the model.
    """
    print(model)
    if hasattr(model, 'model'):
        # For models with a 'model' attribute
        return model.model.layers[layer_idx].mlp
    elif hasattr(model, 'gpt_neox'):
        # For models with a 'gpt_neox' attribute
        return model.gpt_neox.layers[layer_idx].mlp
    elif hasattr(model, 'transformer'):
        # For models with a 'transformer' attribute
        return model.transformer.h[layer_idx].mlp
    else:
        # For models without a 'model' attribute
        return model.transformer.h[layer_idx].mlp

class MultiFileActivationDataLoader(DataLoader):
    def __init__(self, activ_dir, batch_size=1, shuffle=False, shuffle_files=False, verbose=False, check_completed=True):
        self.activ_dir = activ_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_files = shuffle_files
        self.verbose = verbose
        self.files = [f for f in os.listdir(activ_dir) if f.endswith('.pt')]
        if check_completed:
            if not os.path.exists(os.path.join(activ_dir, '.completed')):
                raise ValueError("Activation directory is not completed. Please finish generating.")

        recompute_metadata = False
        metadata_filename = 'length_metadata.pkl'
        metadata_path = os.path.join(activ_dir, metadata_filename)
        if not os.path.exists(metadata_path):
            print(f"Metadata file '{metadata_filename}' not found in {activ_dir}.")
            recompute_metadata = True
        else:
            metadata = pickle.load(open(metadata_path, 'rb'))
            # check if keys of metadata match filenames in activ_dir
            if set(metadata.keys()) != set(self.files):
                print(f"Metadata keys do not match files in {activ_dir}. Recomputing metadata.")
                recompute_metadata = True

        if recompute_metadata:
            print('Computing length naively from files... This may take a while.')
            file_lengths = {}
            for file in tqdm(self.files):
                if file.endswith('.pt'):
                    data = torch.load(os.path.join(activ_dir, file))
                    file_lengths[file] = data.shape[0]
                    del data # Free memory after processing each file
            print('Saving to length_metadata.pkl to avoid recomputing...')
            with open(os.path.join(activ_dir, 'length_metadata.pkl'), 'wb') as f:
                pickle.dump(file_lengths, f)

        length_metadata = pickle.load(open(metadata_path, 'rb'))
        self.length = 0
        for file in self.files:
            self.length += length_metadata[file] // batch_size

        if not self.shuffle_files:
            self.files = sorted(self.files)
        else:
            random.shuffle(self.files)
        self.current_file_index = -1
        self.current_data = None
        self.current_dataloader = None
        self.num_files = len(self.files)

    def load_next_file(self):
        self.current_file_index += 1
        if self.current_file_index < self.num_files:
            if self.verbose:
                print('Loading file:', self.files[self.current_file_index])
            file_path = os.path.join(self.activ_dir, self.files[self.current_file_index])
            self.current_data = torch.load(file_path)
            self.current_dataloader = iter(torch.utils.data.DataLoader(
                self.current_data, batch_size=self.batch_size, shuffle=self.shuffle
            ))
        else:
            if self.verbose:
                print('No more files to load.')
            self.current_data = None
            self.current_dataloader = None

    def reset(self):
        self.current_file_index = -1
        if self.shuffle_files:
            random.shuffle(self.files)
        self.current_data = None
        self.current_dataloader = None

    def __iter__(self):
        return self
    
    def __len__(self):
        return self.length

    def __next__(self):
        if self.current_file_index == -1:
            self.load_next_file()
        while self.current_dataloader is not None:
            try:
                new_batch = next(self.current_dataloader)
                if new_batch.shape[0] != self.batch_size:
                    continue
                else:
                    return new_batch
            except StopIteration:
                self.load_next_file()

        if self.current_dataloader is None:
            raise StopIteration