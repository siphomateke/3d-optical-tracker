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
        self.write_list = [self.server_socket]
        self.open = False

        self.out_buffer = []

    def listen(self):
        self.server_socket.listen(5)
        print "Listening on port {}".format(self.port)
        self.start_thread()

    def send(self, msg, header=""):
        self.out_buffer.append((header, msg))

    def run(self):
        # Fix network select
        readable, writable, errored = select.select(self.read_list, self.write_list, [])
        for sock in readable:
            # On initial connection
            if sock is self.server_socket:
                self.client_socket, address = self.server_socket.accept()
                self.read_list.append(self.client_socket)
                self.write_list.append(self.client_socket)
                print "Connection from", address
                self.open = True

        for client in writable:
            if len(self.out_buffer) > 0:
                try:
                    if len(self.out_buffer[0]) >= 2:
                        combined_str = ""
                        for i in xrange(len(self.out_buffer[0][1])):
                            item = self.out_buffer[0][1][i]
                            combined_str += "{}".format(item)
                            if i < len(self.out_buffer[0][1]) - 1:
                                combined_str += ","

                        if len(combined_str) > 0:
                            self.client_socket.send("{}{}\n".format(self.out_buffer[0][0], combined_str))
                        del self.out_buffer[0]
                except socket.error as err:
                    print "Socket error: {}".format(err)
                    self.write_list.remove(client)
                    self.read_list.remove(client)
                    self.open = False

    def close(self):
        """
        Terminates the network connection and waits for the thread to finish
        """
        if self.open:
            self.client_socket.close()
        self.stop_thread()


if __name__ == '__main__':
    net_socket = NetworkSocket()
    net_socket.listen()
