import torch
from cryptography.fernet import Fernet
import io
import torch.nn as nn
from models.yolo import Detect, Model

class DecryptAndLoadModel:

    def __init__(self, secret_key):
        self.secret_key = secret_key

    def decrypt_file(self, file_path):
        with open(file_path, 'rb') as file:
            encrypted_data = file.read()

        fernet = Fernet(self.secret_key)
        decrypted_data = fernet.decrypt(encrypted_data)

        return decrypted_data

    def load_model(self, decrypted_data):

        model = Ensemble()

        ckpt = torch.load(io.BytesIO(decrypted_data), map_location='cpu')  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(torch.device('cpu')).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if True and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

            # Module compatibility updates
        for m in model.modules():
            t = type(m)
            if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
                m.inplace = True  # torch 1.7.0 compatibility
                if t is Detect and not isinstance(m.anchor_grid, list):
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
            elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        # Return model
        if len(model) == 1:
            return model[-1]

        for k in 'names', 'nc', 'yaml':
            setattr(model, k, getattr(model[0], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
        return model


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output



if __name__ == "__main__":

    secret_key = b'KCilyK8Fk9wHiuEc5PmbZbbcR-ue3VSsqFaoKZRcdvA='

    decryptor = DecryptAndLoadModel(secret_key)

    file_to_decrypt = '/home/helia/internship/task6-file encrypt/decrypt/torch_decrypt/encrypted_data.bin'

    decrypted_data = decryptor.decrypt_file(file_to_decrypt)

    model = decryptor.load_model(decrypted_data)

    model.eval()
