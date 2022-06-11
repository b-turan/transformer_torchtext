def count_parameters(model):
    """ 
    Counts the learnable parameters of a given model.
    ------------------------------------
    model (torch.nn.module): model
    ------------------------------------
    Returns number of learnable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
