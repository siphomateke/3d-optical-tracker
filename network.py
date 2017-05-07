import socket
import select
import math
from threadutil import ProgramThread


class NetworkSocket(ProgramThread):
    def __init__(self, port=35353):
        ProgramThread.__init__(self, self.run)

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setblocking(0)
        self.port = port
        self.server_socket.bind(("", port))
        self.client_socket = None
        self.read_list = [self.server_socket]
        self.open = False

        self.x = 0

    def listen(self):
        self.server_socket.listen(5)
        print "Listening on port {}".format(self.port)
        self.start_thread()

    def run(self):
        readable, writable, errored = select.select(self.read_list, [], [])
        for sock in readable:
            # On initial connection
            if sock is self.server_socket:
                self.client_socket, address = self.server_socket.accept()
                self.read_list.append(self.client_socket)
                print "Connection from", address
                self.open = True

        if self.open:
            y = math.sin(self.x)
            z = math.cos(y)
            self.x += 0.01
            try:
                self.client_socket.send('d:{},{},{}\n'.format(y, y, z))
            except socket.error as err:
                print "Socket error: {}".format(err)
                self.open = False

    def close(self):
        if self.open:
            self.client_socket.close()
        self.shut_down_thread()


"""size = 1024
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setblocking(0)
host = ""
port = 35353
server_socket.bind((host, port))

server_socket.listen(5)
print "Listening on port {}".format(port)

read_list = [server_socket]
x = 0
client_socket = None
while True:
    readable, writable, errored = select.select(read_list, [], [])
    for sock in readable:
        # On initial connection
        if sock is server_socket:
            client_socket, address = server_socket.accept()
            read_list.append(client_socket)
            print "Connection from", address

    if client_socket is not None:
        y = math.sin(x)
        z = math.cos(y)
        x += 0.01
        try:
            client_socket.send('d:{},{},{}\n'.format(y, y, z))
        except socket.error as err:
            print "Socket error: {}".format(err)
            client_socket = None"""

net_socket = NetworkSocket()
net_socket.listen()
