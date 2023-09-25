import base64
import json
import uuid
import enum
from cryptography.hazmat.primitives import serialization, asymmetric
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet



class Action(enum.Enum):
    # Here you will define all the types of packets that you need
    # Ensure to use the same name as String
    # For example:
    Ok = "Ok"
    Sec = "Sec"
    Deny = "Deny"
    Register = "Register"
    Login = "Login"
    Chat = "Chat"
    ModelDelta = "ModelDelta"
    Target = "Target"


class Packet:
    def __init__(self, action: Action, keys, *payloads, no_sec=False):
        print("__init__ = " + str(no_sec))
        self._serer_public_key = keys[0]
        self._client_private_key = keys[1]
        self._client_public_key = keys[2]

        self._id = uuid.uuid4()
        self.action: Action = action
        self.payloads: tuple = payloads
        self.no_sec = no_sec

    @classmethod
    def get_keys(self):
        _server_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        _server_public_key = _server_private_key.public_key()
        
        # Serialize the private and public keys to PEM format
        _server_private_key_text = _server_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        
        _server_public_key_text = _server_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        return _server_private_key, _server_public_key, _server_private_key_text, _server_public_key_text

    def __str__(self):
        data = {"a": str(self.action)}
        for i, payload in enumerate(self.payloads):
            data[f"p{i}"] = payload
        return json.dumps(data)
    
    def __bytes__(self) -> bytes:
        json_string = self.__str__()  # Convert the packet data to a JSON string
        if self.no_sec:
            base64_encrypted_data = json_string
        else:
            encrypted_data = self._client_public_key.encrypt(
                json_string.encode(),
                asymmetric.padding.OAEP(
                    mgf=asymmetric.padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            base64_encrypted_data = base64.b64encode(encrypted_data).decode('utf-8')
        return base64_encrypted_data  # Return as base64 string

def from_json(base64_str, client_public_key, no_sec=False):
    print("from_json = " + str(no_sec))
    if no_sec:
        json_str = base64_str
    else:
        encrypted_data = base64.b64decode(base64_str)  # Decode base64 string to bytes
        print("OK1")
        decrypted_data = client_public_key.decrypt(
            encrypted_data,
            asymmetric.padding.OAEP(
                mgf=asymmetric.padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        print("OK2")
        json_str = decrypted_data.decode('utf-8')  # Convert bytes to string

    print("OK3")
    data = json.loads(json_str)
    print("OK4")
    
    action = None
    payloads = []
    for key, value in data.items():
        if key == "id":
            id = value
        elif key == "a":
            action = value
        elif key.startswith("p"):
            payloads.append(value)
    
    print("action = " + action)
    class_name = f"{action.replace('Action.', '').capitalize()}Packet"
    print("class_name = " + class_name)
    try:
        
        cls = globals().get(class_name)
        print("cls")
        if cls:
            print("return cls")
            return cls(*payloads)
        else:
            return Packet(action, *payloads)
    except KeyError as e:
        print("OK")
    except TypeError as e:
        print("OK")

# Here you would define other packet types as subclasses of Packet
# For example:


class SecPacket(Packet):
    def __init__(self, keys, no_sec=False):
        print("SecPacket = " + str(no_sec))
        super().__init__(Action.Sec, keys, no_sec=no_sec)

class DenyPacket(Packet):
    def __init__(self, keys, reason: str, no_sec=False):
        super().__init__(Action.Deny, keys, reason, no_sec=no_sec)

class OkPacket(Packet):
    def __init__(self, keys, no_sec=False):
        super().__init__(Action.Ok, keys, no_sec=no_sec)

class RegisterPacket(Packet):
    def __init__(self, keys,username: str, password: str, email: str, avatar_id: int, no_sec=False):
        super().__init__(Action.Register, keys, username, password, email, avatar_id, no_sec=no_sec)

class LoginPacket(Packet):
    def __init__(self, keys, username: str, password: str, no_sec=False):
        super().__init__(Action.Login, keys, username, password, no_sec=no_sec)

class ChatPacket(Packet):
    def __init__(self, keys, sender: str, message: str, no_sec=False):
        super().__init__(Action.Chat, keys, sender, message, no_sec=no_sec)

class ModelDeltaPacket(Packet):
    def __init__(self, keys, model_data: dict, no_sec=False):
        super().__init__(Action.ModelDelta, keys, model_data, no_sec=no_sec)

class TargetPacket(Packet):
    def __init__(self, keys, t_x: float, t_y: float, no_sec=False):
        super().__init__(Action.Target, keys, t_x, t_y, no_sec=no_sec)