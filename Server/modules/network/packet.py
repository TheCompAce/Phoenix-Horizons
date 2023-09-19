import json
import uuid
import enum
import enum

class Action(enum.Enum):
    # Here you will define all the types of packets that you need
    # Ensure to use the same name as String
    # For example:
    Ok = "Ok"
    Deny = "Deny"
    Register = "Register"
    Login = "Login"
    Chat = "Chat"
    ModelDelta = "ModelDelta"
    Target = "Target"

class Packet:
    def __init__(self, action: Action, *payloads):
        self._id = uuid.uuid4()
        self.action: Action = action
        self.payloads: tuple = payloads

    def __str__(self):
        data = {"a": str(self.action)}
        for i, payload in enumerate(self.payloads):
            data[f"p{i}"] = payload
        return json.dumps(data)
    
    def __bytes__(self) -> bytes:
        return str(self).encode('utf-8')

def from_json(json_str):
    print("json_str = " + json_str)
    id = None
    data = json.loads(json_str)
    
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
class ChatPacket(Packet):
    def __init__(self, sender: str, message: str):
        super().__init__(Action.Chat, sender, message)

class OkPacket(Packet):
    def __init__(self):
        super().__init__(Action.Ok)

class DenyPacket(Packet):
    def __init__(self, reason: str):
        super().__init__(Action.Deny, reason)

class RegisterPacket(Packet):
    def __init__(self, username: str, password: str, email: str, avatar_id: int):
        super().__init__(Action.Register, username, password, email, avatar_id)

class LoginPacket(Packet):
    def __init__(self, username: str, password: str):
        super().__init__(Action.Login, username, password)

class ModelDeltaPacket(Packet):
    def __init__(self, model_data: dict):
        super().__init__(Action.ModelDelta, model_data)

class TargetPacket(Packet):
    def __init__(self, t_x: float, t_y: float):
        super().__init__(Action.Target, t_x, t_y)