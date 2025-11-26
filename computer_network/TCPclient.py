from socket import *
serversocket = 'localhost'
serverport = 1200
clientsocket = socket(AF_INET,SOCK_STREAM)
clientsocket.connect((serversocket,serverport))

sentence = input('Input lowercase sentence:')
clientsocket.send(sentence.encode())

modifiedsentence = clientsocket.recv(1024)
print('From server:',modifiedsentence.decode())
clientsocket.close()