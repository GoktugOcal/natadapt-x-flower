from time import sleep

def transfer(message):
    #sleep(5)
    print(message)


def grpc_channel_manager():
    try:
        # Some setup code for the gRPC channel
        receive = transfer
        send = transfer
        yield (receive, send)
    finally:
        # Cleanup code, always executed
        print("DEBUG :: gRPC channel closed")


gen = grpc_channel_manager()
receive_channel, send_channel = next(gen)
for i in range(10):
    receive_channel("rec")
    send_channel("send")
    if i == 5:
        break

from typing import Callable
from queue import Queue
class ClientMessage:
    print("Client message...")
    pass

queue = Queue()  # Some queue instance

send: Callable[[ClientMessage], None] = lambda msg: queue.put(msg, block=False)

# Now you can use the send function to put messages into the queue
message = ClientMessage()
print("message")
send(message)
print("end")
