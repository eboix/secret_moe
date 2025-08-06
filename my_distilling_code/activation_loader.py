# activation_loader.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import warnings
from utils import extract_layer_mlp

class ActivationDataLoader(IterableDataset):
    """
    A DataLoader-like class that extracts activations from a specified layer of a model.

    Args:
        model (nn.Module): The Hugging Face model.
        tokenizer: The tokenizer associated with the model.
        layer_idx (int): The index of the layer from which to extract activations.
                         For many transformer models, this refers to an index in
                         `model.model.layers` or `model.transformer.h` or `model.gpt_neox.layers`.
        text_dataloader (DataLoader): A DataLoader that yields batches of raw text strings.
        activation_type (str): 'input' or 'output' of the layer.
        max_length (int, optional): Maximum sequence length for tokenization. 
                                    If None, uses tokenizer.model_max_length. Defaults to 512 if tokenizer has no default.
        batch_size (int): Number of activation vectors (tokens) to yield in each batch from this loader.
        max_buffer_size (int): Maximum number of activation vectors to store in the internal buffer
                               before yielding.
        dtype (torch.dtype): The desired data type for the activations.
        device (str or torch.device): The device to run the model on ('cuda', 'cpu').
    """
    def __init__(self, model, tokenizer, layer_idx, text_dataloader, 
                 activation_type='input', max_length=None, batch_size=32, 
                 max_buffer_size=100000, dtype=torch.float32, device='cuda'):
        self.model = model.to(device)
        self.model.eval() # Ensure model is in eval mode
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.text_dataloader = text_dataloader # DataLoader yielding batches of raw text strings
        self.activation_type = activation_type.lower()
        
        if max_length is None:
            self.max_length = tokenizer.model_max_length if tokenizer.model_max_length is not None else 512
            if self.max_length > 2048: # Safety cap for some models without good defaults
                warnings.warn(f"Tokenizer model_max_length is {self.max_length}. Capping to 2048 for ActivationDataLoader.")
                self.max_length = 2048
        else:
            self.max_length = max_length

        self.batch_size = batch_size # This is the batch size of *activation vectors* to yield
        self.max_buffer_size = max_buffer_size
        self.dtype = dtype
        self.device = device

        self.buffer = [] # Stores (seq_len * hidden_dim) tensors
        self.text_iter = iter(self.text_dataloader) # Iterator for the raw text
        
        self._hook_handle = None
        self._activation_cache = [] # Temporarily stores activations from a single forward pass

        self._register_hook()

    def _hook_fn(self, module, input_act, output_act):
        """Internal hook function to capture activations."""
        self.model.eval() # Ensure model stays in eval mode
        act_to_store = None
        if self.activation_type == 'input':
            # Input to a transformer layer is often a tuple (hidden_states, attention_mask, ...)
            # We typically want the first element: hidden_states
            if isinstance(input_act, tuple) and len(input_act) > 0:
                act_to_store = input_act[0].detach()
            else:
                assert(False), "Input activation should be a tuple"

        elif self.activation_type == 'output':
            # Output of a transformer layer can also be a tuple (hidden_states, ...)
            if isinstance(output_act, tuple) and len(output_act) > 0:
                act_to_store = output_act[0].detach()
            else:
                # If output is a single tensor
                act_to_store = output_act.detach()
        else:
            if self._hook_handle: self._hook_handle.remove() # Clean up hook
            raise ValueError(f"activation_type must be 'input' or 'output', got {self.activation_type}")
        
        if act_to_store is not None:
            # act_to_store shape is (batch_size_text, seq_len, hidden_dim)
            # We want to store individual token activations: (batch_size_text * seq_len, hidden_dim)
            act_reshaped = act_to_store.reshape(-1, act_to_store.shape[-1]).to(dtype=self.dtype)
            self._activation_cache.append(act_reshaped)
        else:
            warnings.warn("No activation captured in hook function. Check if the layer index is correct or if the model is in eval mode.")
            assert(False)


    def _find_target_layer(self):
        return extract_layer_mlp(self.model, self.layer_idx)

    def _register_hook(self):
        """Registers the forward hook to the target layer."""
        if self._hook_handle is not None: # Remove existing hook if any
            self._hook_handle.remove()
            
        target_layer = self._find_target_layer()
        if target_layer is None: # Should be caught by _find_target_layer raising an error
            raise ValueError(f"Target layer {self.layer_idx} not found.")
            
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
        if self._hook_handle is None:
            raise RuntimeError(f"Failed to register hook for layer {self.layer_idx}.")


    def _fill_buffer(self):
        """Fills the internal buffer with activations up to max_buffer_size."""
        is_text_exhausted = False
        while len(self.buffer) < self.max_buffer_size:
            try:
                # Get a batch of raw text strings from the text_dataloader
                # The collate_fn in text_dataloader should prepare a list of strings
                raw_text_list = next(self.text_iter) 
                
                if not raw_text_list: # Skip if batch is empty
                    continue

                # Tokenize the batch of text
                # The text_dataloader's batch_size determines how many texts are tokenized at once
                inputs = self.tokenizer(raw_text_list, return_tensors="pt", 
                                        padding=True, truncation=True, 
                                        max_length=self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                self._activation_cache = [] # Clear cache for this new forward pass
                with torch.no_grad():
                    self.model(**inputs) # Forward pass to trigger hook
                
                if self._activation_cache: # If hook captured something
                    # Concatenate activations from this pass (usually just one tensor)
                    # and add individual activation vectors to the main buffer
                    current_pass_activations = torch.cat(self._activation_cache, dim=0)
                    self.buffer.extend(current_pass_activations) # Add tensor rows
                self._activation_cache = [] # Clear again

            except StopIteration: # text_dataloader is exhausted
                is_text_exhausted = True
                break # Exit the while loop
            except Exception as e:
                warnings.warn(f"Error during _fill_buffer tokenization or model forward pass: {e}")
                # Depending on the error, you might want to skip this batch or stop
                continue # Skip this batch and try next

        return is_text_exhausted # True if text_dataloader is exhausted

    def __iter__(self):
        # Reset / Re-register hook if needed, though typically done at init
        # self._register_hook() 
        self.text_iter = iter(self.text_dataloader) # Ensure text iterator is fresh
        self.buffer = [] # Clear buffer for new iteration
        return self

    def __next__(self):
        if not self.buffer: # If buffer is empty, try to fill it
            text_exhausted = self._fill_buffer()
            if text_exhausted and not self.buffer: # If text is done and buffer is still empty
                self._cleanup_hook()
                raise StopIteration

        if self.buffer:
            # Yield a batch of activation vectors of size self.batch_size
            num_to_yield = min(self.batch_size, len(self.buffer))
            
            # Stack the collected 1D activation tensors into a 2D batch tensor
            activations_batch_list = self.buffer[:num_to_yield]
            activations_batch_tensor = torch.stack(activations_batch_list)

            self.buffer = self.buffer[num_to_yield:] # Remove yielded items from buffer
            return activations_batch_tensor.cpu() # Move to CPU before returning
        else: 
            # This case should ideally be caught by the StopIteration above
            self._cleanup_hook()
            raise StopIteration
            
    def _cleanup_hook(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self):
        # Ensure hook is removed when object is deleted
        self._cleanup_hook()