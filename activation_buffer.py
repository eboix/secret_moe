import random
import torch
import numpy as np
import transformers
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# Define a DataLoader class for extracting activations from transformer models
# Given a text_dataloader, this class extracts MLP activations from a specified layer of the model.
# Internally, activations are stored in a buffer and yielded in batches.
# The batches yield a random subset of the activations. When the buffer is half full, it is replenished with new activations.
class ActivationDataLoader:
    def __init__(self, model, tokenizer, layer_idx=1, device=None, max_length=None, batch_size=32, 
                 activation_type='input', shuffle=True, text_dataloader=None,
                 max_buffer_size=100000):
        """
        Initialize the data loader for extracting MLP activations from a transformer model.
        
        Args:
            model: The transformer model to extract activations from
            tokenizer: The tokenizer for the model
            layer_idx: Which layer to extract activations from (default: 1)
            device: Device to run the model on (default: current device of model)
            max_length: Maximum sequence length (default: None)
            batch_size: Size of batches to process (default: 32)
            activation_type: Type of activations to extract ('input', 'output', or 'both')
            shuffle: Whether to shuffle the dataset (default: True)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device if device is not None else model.device
        self.max_length = max_length
        self.batch_size = batch_size
        self.activation_type = activation_type
        self.shuffle = shuffle
        self.text_dataloader = text_dataloader
        self.text_iterator = iter(text_dataloader)

        if type(model) == transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM:
            self.activation_dim = model.gpt_neox.layers[layer_idx].mlp.dense_h_to_4h.in_features
        else:
            raise NotImplementedError("Only GPT-NeoX models are supported in this function.")

        self.max_buffer_size = max_buffer_size
        self.activation_buffer = torch.zeros((max_buffer_size, self.activation_dim), device=self.device)
        self.read = torch.ones(max_buffer_size, dtype=torch.bool, device=self.device)
        self.num_in_buffer = 0
    
    def extract_activations(self, text_batch):
        """Extract activations from the model for a batch of text."""
        # Tokenize the input text
        inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True, 
                               truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Store activations
        activations = []
        
        # Hook function to capture activations
        def hook_fn(module, input, output):
            if self.activation_type == 'input':
                # Store the input activations
                activations.append(input[0].view(-1, self.activation_dim))
            elif self.activation_type == 'output':
                # Store the output activations
                activations.append(output.view(-1, self.activation_dim))
            elif self.activation_type == 'both':
                # Store both input and output activations
                activations.append((input[0].view(-1, self.activation_dim), output.view(-1, self.activation_dim)))
            else:
                raise ValueError("activation_type must be 'input', 'output', or 'both'.")
        
        # Register the hook on the desired layer's MLP
        if isinstance(self.model, transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXForCausalLM):
            hook = self.model.gpt_neox.layers[self.layer_idx].mlp.register_forward_hook(hook_fn)
        else:
            raise NotImplementedError("Only GPT-NeoX models are supported in this function.")
        
        # Forward pass with no gradient computation
        with torch.no_grad():
            self.model(**inputs)
        
        # Remove the hook
        hook.remove()
        
        return activations[0]
    

    def _get_next_text(self):
        assert(self.text_dataloader.batch_size == 1)
        text = next(self.text_iterator, None)[0]
        if text is None:
            print("No more text samples available in the dataloader.")
            print('Refreshing iterator...')
            self.text_iterator = iter(self.text_dataloader)
            text = next(self.text_iterator)[0]
        while len(text) < 20:
            text = next(self.text_iterator)[0]
        return text
    
    def replenish_buffer(self):
        """Replenish the activation buffer with new activations."""
        # Collect activations from the text dataloader

        print('Replenishing buffer...')
        current_buffer_size = self.num_in_buffer
        self.activation_buffer[:current_buffer_size,:] = self.activation_buffer[~self.read,:]
        
        new_activation_buffer = []
        new_buffer_len = 0

        remaining = self.max_buffer_size - current_buffer_size
        with tqdm(total=remaining, desc="Replenishing activations") as pbar:
            while new_buffer_len + current_buffer_size < self.max_buffer_size:
                text = self._get_next_text()
                activations = self.extract_activations([text])
                new_activation_buffer.append(activations)
                increment = activations.shape[0]
                new_buffer_len += increment
                pbar.update(increment)

        new_activation_buffer = torch.vstack(new_activation_buffer)
        self.activation_buffer[current_buffer_size:,:] = new_activation_buffer[:self.max_buffer_size - current_buffer_size, :]
        # set all read to False
        self.read[:] = False
        self.num_in_buffer = self.max_buffer_size
        print('Activation buffer replenished. Current size:', self.num_in_buffer)
        

    def __iter__(self):
        """Iterator that yields batches of activations."""
        assert(self.batch_size < self.max_buffer_size / 2)
        while True:
            if self.num_in_buffer < self.max_buffer_size / 2:
                self.replenish_buffer()

            # Select self.batch_size random indices from the buffer that are not currently read
            indices = torch.where(~self.read)[0]
            selected_indices = random.sample(indices.tolist(), self.batch_size)
            # Mark these indices as read
            self.read[selected_indices] = True
            # Yield the activations for these indices
            activations = self.activation_buffer[selected_indices, :]
            self.num_in_buffer -= self.batch_size
            yield activations