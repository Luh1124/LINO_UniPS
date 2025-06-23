import torch
from src.models import LiNo_UniPS 
from src.data import TestData
from src.data import DemoData
dependencies = ['torch', 'pytorch_lightning', 'numpy']

DEFAULT_MODEL_URL = "https://huggingface.co/houyuanchen/lino/resolve/main/lino.pth"  

def lino_unips(pretrained=True, task_name="DiLiGenT", **kwargs):
    model = LiNo_UniPS(task_name=task_name, **kwargs)
    if pretrained:
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                DEFAULT_MODEL_URL,
                progress=True
            )
            model.load_state_dict(state_dict) #
            model.eval()
            print("load lino_unips successfully")
        except Exception as e:
            print(f"error{e}")
            
    return model

def load_test_data(data_root: list, numofimages: int):
    return TestData(data_root,numofimages)

def load_data(input_imgs_list, input_mask):
    return DemoData(input_imgs_list,input_mask)