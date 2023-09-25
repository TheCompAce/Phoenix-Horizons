import base64
import json
from queue import Queue
import uuid
import time
import math

from autobahn.twisted.websocket import WebSocketServerProtocol
from autobahn.exception import Disconnected

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from modules.network import packet
from modules.database.worker import User, Base, Entity, InstancedEntity, Actor
from modules.tools.ph_math import direction_to

server_private_key, server_public_key, server_private_key_text, server_public_key_text = packet.Packet.get_keys()
client_private_key, client_public_key, client_private_key_text, client_public_key_text = packet.Packet.get_keys()
# packet.Packet.set_server_keys(server_private_key(), server_public_key())

engine = create_engine('sqlite:///./server.db')
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

class GameServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super().__init__()
        self._uuid = uuid.uuid4()
        self._packet_queue: Queue[tuple['GameServerProtocol', packet.Packet]] = Queue()
        self._state: callable = None
        self._actor: Actor = None
        self._player_target: list[float] = None
        self._last_delta_time_checked: float = None
        self._known_others: set['GameServerProtocol'] = set()
        self.keys = [server_public_key, client_private_key, client_public_key]

    def broadcast(self, p: packet.Packet, exclude_self: bool = False):
        for other in self.factory.players:
            if other == self and exclude_self:
                continue
        
            other.onPacket(self, self.keys, p)

    def onPacket(self, sender: 'GameServerProtocol', p: packet.Packet):
        self._packet_queue.put((sender, p))
        print(f"Queued packet: {p}")

    def onConnect(self, request):
        print(f"Client connecting: {request.peer}")

    def onOpen(self):
        print(f"Websocket connection open.")
        keys = {
                "spk": server_public_key_text,
                "crk":  client_private_key_text,
                "cpk": client_public_key_text
            }
        keys = json.dumps(keys)
        keys = bytes(keys, 'utf-8')
        try:
            self.send_client(keys, True)
            self._state = self.SECURITY
        except Disconnected as e:
            print(f"Couldn't send keys because client disconnected.")

        

    def onMessage(self, payload, isBinary=False, no_sec=False):
        decoded_payload = payload.decode('utf-8')
        print(f"Received message: {decoded_payload}")
        if no_sec:
            data = decoded_payload
        else:
            data = base64.b64decode(decoded_payload)

        print(data)
        
        try:
            print("decoded_payload = " + data)
            p: packet.Packet = packet.from_json(data, client_public_key, no_sec)
            self.onPacket(self, self.keys, p)
        except Exception as e:
            print(f"Could not load message as packet: {e}. Message was: {payload.decode('utf-8')}")

    def onClose(self, wasClean, code, reason):
        if self._actor:
            self._actor.save(session)

        self.factory.players.remove(self)
        print(f"Websocket connection closed {' unxpectedly' if not wasClean else 'cleanly'} with code {code}: {reason}")        

    def send_client(self, p: packet.Packet, no_sec=False):
        b = bytes(p)
        try:
            self.sendMessage(b, no_sec=no_sec)
        except Disconnected as e:
            print(f"Couldn't send {p} because client disconnected.")

    # Handle Tick Actions
    def tick(self):
        if not self._packet_queue.empty():
            s,p = self._packet_queue.get()
            self._state(s, p)
        elif self._state == self.PLAY:
            actor_dict_before: dict= self._actor.to_dict(session)
            if self._update_position(session):
                actor_dict_after: dict= self._actor.to_dict(session)
                self.broadcast(packet.ModelDeltaPacket(Actor.to_delta_dict(actor_dict_before, actor_dict_after)))

    # Functions for Custom calls
    def SECURITY(self, sender: 'GameServerProtocol', p: packet.Packet):
        print("SECURITY")
        if p.action == packet.Action.Sec:
            chk_key = p.payloads
            if chk_key == self.client_public_key:
                self._state = self.LOGIN
            

    def LOGIN(self, sender: 'GameServerProtocol', p: packet.Packet):
        if p.action == packet.Action.Login:
            username, password = p.payloads
            #user = session.query(User).filter(User.username == username, User.password == password).first()
            user = User.query(session, filter_by={'username': username}, first=True, exclude={'User': ['password', 'email']})
            if user and user.verify_password(password):
                user = user
                #self._actor = session.query(Actor).filter(Actor.user == user).first()
                self._actor = Actor.query(session, filter_by={'user_id': user.id}, first=True, exclude={'User': ['password', 'email']})
                self.send_client(packet.OkPacket(self.keys))
                self.broadcast(packet.ModelDeltaPacket(self._actor.to_dict(session)))
                self._state = self.PLAY
            else:
                self.send_client(packet.DenyPacket(self.keys, "Username or Password is incorrect."))
        elif p.action == packet.Action.Register:
            username, password, email, avatar_id = p.payloads
            
            #user = session.query(User).filter(User.username == username).first()
            user = User.query(session, filter_by={'username': username}, first=True, exclude={'User': ['password', 'email']})
            if user:
                self.send_client(packet.DenyPacket(self.keys, "This username is taken."))
            else:
                user = User(username=username, password=password, email=email)
                player_entity = Entity(name=username)
                player_ientity = InstancedEntity(entity=player_entity, x=0, y=0)
                player = Actor(instanced_entity=player_ientity, user=user, avatar_id=avatar_id)

                user.save(session)
                player_entity.save(session)
                player_ientity.save(session)
                player.save(session)

                self.send_client(packet.OkPacket(self.keys))
        else:
            print(f"Received invalid action: {p.action} with payloads: {p.payloads}")

    def PLAY(self, sender: 'GameServerProtocol', p: packet.Packet):
        if p.action == packet.Action.Chat:
            if sender == self:
                self.broadcast(p, exclude_self=True)
            else:
                self.send_client(self.keys, p)
        elif p.action == packet.Action.Target:
            self._player_target = p.payloads
        elif p.action == packet.Action.ModelDelta:
            self.send_client(self.keys, p)

            if sender not in self._known_others:
                sender.onPacket(self, packet.ModelDeltaPacket(self.keys, self._actor.to_dict(session)))
                self._known_others.add(sender)
        else:
            print(f"Received invalid action: {p.action} with payloads: {p.payloads}")

    def _update_position(self, session) -> bool:
        if not self._player_target:
            return False
        
        self._actor = session.merge(self._actor)
        pos = [self._actor.instanced_entity.x, self._actor.instanced_entity.y]
        now: float = time.time()
        delta_time: float = 1 / self.factory.tickrate
        if (self._last_delta_time_checked):
            delta_time = now - self._last_delta_time_checked
        
        self._last_delta_time_checked = now

        dist: float = 70 * delta_time

        if (math.dist(pos, self._player_target)) < dist:
            return False
        
        d_x, d_y = direction_to(pos, self._player_target)
        self._actor.instanced_entity.x += d_x * dist
        self._actor.instanced_entity.y += d_y * dist

        self._actor.instanced_entity.save(session)

        return True
