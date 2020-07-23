import torch
from collections import OrderedDict
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

def load_model(model, model_path):
    '''
    Accurate model loading function: if model was saved DataParallel mode
    it removes 'module' terms from state_dict
    :param model:
    :param model_path:
    :return:
    '''
    loaded_state_dict = torch.load(model_path, map_location='cpu')
    load_state_dict(model, loaded_state_dict)


def load_state_dict(model, loaded_state_dict):
    '''
    Accurate model loading function: if model was saved DataParallel mode
    it removes 'module' terms from state_dict
    :param model:
    :param loaded_state_dict:
    :return:
    '''

    model_state_dict = model.state_dict()

    new_state_dict = OrderedDict()
    for k, v in loaded_state_dict.items():
        # if DataParallel was used
        name = k.replace("module.", "")
        if v.shape != model_state_dict[name].shape:
            continue
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict,  strict=False)


def freeze_layers(model, layer_type):
    for m in model.modules():
        if not isinstance(m, layer_type):
            m.requires_grad_(requires_grad=False)
        else:
            m.requires_grad_()


def unfreeze_layers(model):
    frozen_parameters = filter(lambda p: p.requires_grad == False, model.parameters())
    for param in frozen_parameters:
        param.requires_grad = True
    return frozen_parameters