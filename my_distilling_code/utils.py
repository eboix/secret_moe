
def extract_layer_mlp(model, layer_idx):
    """
    Extract the MLP layer from the specified layer index of the model.
    """
    if hasattr(model, 'model'):
        # For models with a 'model' attribute
        return model.model.layers[layer_idx].mlp
    elif hasattr(model, 'gpt_neox'):
        # For models with a 'gpt_neox' attribute
        return model.gpt_neox.layers[layer_idx].mlp
    else:
        # For models without a 'model' attribute
        return model.transformer.h[layer_idx].mlp