from socket import *
serverport = 1200
serversocket = socket(AF_INET,SOCK_DGRAM)
serversocket.bind(('',serverport))

print("the server is ready to receive")
while True:
    message, clientaddress = serversocket.recvfrom(2048)
    modifiedmessage = message.decode().upper()
    serversocket.sendto(modifiedmessage.encode(),clientaddress)