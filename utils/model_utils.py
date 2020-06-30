import torch

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def weights_init(m):
    """
    # custom weights initialization
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        # torch.nn.init.normal_(m.weight, 1.0, 0.02)
        # torch.nn.init.zeros_(m.bias)
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
