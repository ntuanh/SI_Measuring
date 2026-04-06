import sys , yaml

from src.transmission import Transmission

arg = sys.argv[1]

with open('config.yaml') as file:
    config = yaml.safe_load(file)

trans = Transmission(config=config)

print("arg " , arg)

if arg == "server":
    print("Running Server !")
    trans.server()

elif arg == "edge":
    print("Running Edge !")
    trans.edge()
elif arg == "cloud":
    print("Running Cloud !")
    trans.cloud()
else :
    print("Wrong argument ")

