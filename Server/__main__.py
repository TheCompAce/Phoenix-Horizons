import json
import sys
import uuid

from sqlalchemy import create_engine
from modules.network import protocol
from twisted.python import log
from twisted.internet import reactor, task, ssl
from autobahn.twisted.websocket import WebSocketServerFactory
from modules.database.worker import User, Base, init

init()

class GameFactory(WebSocketServerFactory):
    def __init__(self, hostname: str, port: int):
        self.protocol = protocol.GameServerProtocol
        super().__init__(f"wss://{hostname}:{port}")

        self._uuid = uuid.uuid4()
        self._ticks = 0
        self.players: set[protocol.GameServerProtocol] = set()
        self.tickrate: int = 20

        ticklooop = task.LoopingCall(self.tick)
        ticklooop.start(1 / self.tickrate)

    def tick(self):
        self._ticks += 1
        for p in self.players:
            p.tick()

    #Override
    def buildProtocol(self, addr):
        p = super().buildProtocol(addr)
        self.players.add(p)
        return p

if __name__ == '__main__':
    log.startLogging(sys.stdout)
    engine = create_engine('sqlite:///./server.db')
    Base.metadata.create_all(bind=engine)

    try:
        with open('server.json', 'r') as file:
            data = json.load(file)
            address = data.get('address', '0.0.0.0')
            port = data.get('port', 6438)
    
    except FileNotFoundError:
        print("server.json file not found, using default values")
        address = '0.0.0.0'
        port = 6438



    certs_dir: str = f"{sys.path[0]}/certs/"
    contextFactory = ssl.DefaultOpenSSLContextFactory(certs_dir + "server.key", certs_dir + "server.crt")

    factory = GameFactory(address, port)

    reactor.listenSSL(port, factory, contextFactory)
    reactor.run()