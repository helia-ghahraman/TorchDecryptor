# torchDecryptor

This project will decrypt and load the encrypted AI-model file and return the 'ModelProto' and 'InferenceSession'.

<font color="red">***note***</font>: You must have the secret key to your encrypted file hard-coded in the project.

---
## Author

- [Helia_Ghahraman](https://github.com/helia-ghahraman)


---
## Installation

For installing the dependencies, just run the bellow command:
```bash
  pip install -r requirement.txt

```

### Get the project
```bash
  git clone https://github.com/helia-ghahraman/torchDecryptor.git
  cd torchDecryptor
```
 
---
## API Reference

#### Import Classes And Create Objects

```python
import torch
from cryptography.fernet import Fernet
import io
import torch.nn as nn
from models.yolo import Detect, Model

```

```python

    decryptor = DecryptAndLoadModel(secret_key)

    file_to_decrypt = 'encrypted_data.bin'

    decrypted_data = decryptor.decrypt_file(file_to_decrypt)

    model = decryptor.load_model(decrypted_data)


```

| Parameter               | Type                   | Description                                                                                                                                |
|:------------------------|:-----------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| `file_to_decrypt`       | `int`                  | holds the encrypted file's absolute path.                                                                                                  |
| `decryptor`             | `DecryptAndLoadModel`  | It is an object we create in order to use its functions.                                                                                   |
| `decrypted_data`        | `bytes`                | This parameter holds the bytes of the decrypted data.                                                                                      |
| `model`                 | `ModelProto`           | This parameter holds the the final ModelProto object that we needed.                                                                       |

<br> 
<br>

### DecryptAndLoadModel Class
This class contains 2 methods called **decrypt_file** and **load_model**.

#### Decrypt_file
This function takes a path to the encrypted file as an argument and decrypts the data and returns it.

```python
class DecryptAndLoadModel:

    def decrypt_file(self, file_path):

        return decrypted_data
```
| Function name                  | Return type        | Description                                                                                                                               |
|:----------------------------|:-------------------|:------------------------------------------------------------------------------------------------------------------------------------------|
| `decrypt_file`              | `bytes`            | Decrypts the data and returns its bytes.                                                                                                  |

<br>

#### Load_model
This function takes decrypted data as an argumant and returns the model.

```python
    def load_model(self, decrypted_data):

        return model
```
| Function name                  | Return type        | Description                                                                                                                               |
|:----------------------------|:-------------------|:------------------------------------------------------------------------------------------------------------------------------------------|
| `load_model`                | `ModelProto`       | Returns the onnx AI model.                                                                                                                |

<br>
<br>
