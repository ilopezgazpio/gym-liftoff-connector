import socket

def init_udp_socket():

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sock.bind(("127.0.0.1", 9001))

    return sock