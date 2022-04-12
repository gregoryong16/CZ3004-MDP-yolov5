import bluetooth, subprocess
nearby_devices = bluetooth.discover_devices(duration=10,lookup_names=True,
                                                      flush_cache=True, lookup_class=False)
name = "OPPO Reno5 5G"     # Device name
addr = "14:5E:69:16:C0:06"     # Device Address
port = 1        # RFCOMM port
passkey = "1111" # passkey of the device you want to connect

# kill any "bluetooth-agent" process that is already running
subprocess.call("kill -9 `pidof bluetooth-agent`",shell=True)

# Start a new "bluetooth-agent" process where XXXX is the passkey
status = subprocess.call("bluetooth-agent " + passkey + " &",shell=True)

# Now, connect in the same way as always with PyBlueZ
try:
    s = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    s.connect((addr,port))
except bluetooth.btcommon.BluetoothError as err:
    # Error handler
    pass

#s.recv(1024) # Buffer size
#s.send("Hello World!")

## alternative 
#import socket
#s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
#s.connect(('B8:27:EB:22:57:E0', 1))
#s.send(b'Hello')
#s.recv(1024)
#s.close()