import sys
from ale_python_interface import ALEInterface
from parameters import shared, constants
from Agent import AgentThread
import network
import tensorflow as tf
import net_compute
import threading

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

    critic_weights  = net_compute.WeightHolder(nb_actions, "critic_holder") 
    network_weights = net_compute.WeightHolder(nb_actions, "network_holder")

    agent_pool = []
    tlock = threading.Lock()
    rlock = threading.Lock()

    barrier = threading.Barrier(constants.nb_agent)

    config = tf.ConfigProto(device_count = {"CPU" : constants.nb_agent},
                            inter_op_parallelism_threads = 1,
                            intra_op_parallelism_threads = 1)

    session = tf.Session(config = config)

    for i in range(0, constants.nb_thread):
        agent_pool.append(AgentThread(  network_weights, 
                                        critic_weights,
                                        session,
                                        tlock,
                                        rlock,
                                        barrier,
                                        i
                                    ))
    session.run(tf.initialize_all_variables())

    for agent in agent_pool:
        agent.start()

    for agent in agent_pool:
        agent.join()

    network_weights.save(session, 'network_weights.bck.tf')
    
if __name__ == '__main__':
    main()
