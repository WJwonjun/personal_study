from socket import *
servername = 'localhost'
serverport = 1200
clientsocket = socket(AF_INET, SOCK_DGRAM)

message = input('Input lowercase sentence:')
clientsocket.sendto(message.encode(),(servername,serverport))

modified_message, serveraddress = clientsocket.recvfrom(2048)
print(modified_message.decode())

clientsocket.close()