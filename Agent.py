import threading
from parameters import shared
from ale_python_interface import ALEInterface
from random import randrange
from network import AgentSubNet

class AgentThread(threading.Thread):
    def __init__(self, lock):
        threading.Thread.__init__(self)
        self.lock = lock
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', randrange(0,256,1))
        self.ale.loadROM(shared.game_name)
        self.network = AgentSubNet()
        self.t = 0
        print ("Youhouuuuu")

    def run(self):
       return  
