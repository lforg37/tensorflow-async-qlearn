import Agent

def main():
    if len(sys.argv) < 2:
        print("Missing rom name !")
        return
    romname = sys.argv[1].encode('ascii')
    lock = threading.Lock()
    for i in range(1,3):
        thread = AgentThread(lock)
        thread.start()
    
if __name__ == '__main__':
    main()
