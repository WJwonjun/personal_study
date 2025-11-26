from socket import *
serverport = 1200
serversocket = socket(AF_INET,SOCK_STREAM)
serversocket.bind(('',serverport))
serversocket.listen(1)
print('the server is ready to receive')

while True:
    connectionsocket, addr = serversocket.accept()
    sentence = connectionsocket.recv(1024).decode()
    capsentence = sentence.upper()
    connectionsocket.send(capsentence.encode())
    connectionsocket.close()
