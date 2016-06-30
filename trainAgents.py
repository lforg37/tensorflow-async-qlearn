import sys
from ale_python_interface import ALEInterface
from parameters import shared, constants
from Agent import AgentThread
import network
from threading import Lock

def main():
    if len(sys.argv) < 2:
        print("Missing rom name !")
        return

    # Récupération du nombre d'actions
    romname = sys.argv[1].encode('ascii')
    shared.game_name = romname
    ale = ALEInterface()
    ale.loadROM(romname)
    nb_actions = len(ale.getMinimalActionSet())
    shared.nb_actions = nb_actions
    network.createGlobalWeights()
    agent_pool = []
    lock = Lock()

    for _ in range(0, constants.nb_thread):
        agent_pool.append(AgentThread(lock))

    
if __name__ == '__main__':
    main()
