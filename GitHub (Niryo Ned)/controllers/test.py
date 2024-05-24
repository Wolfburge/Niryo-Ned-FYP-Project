from controller import Supervisor
robot = Supervisor()
can = robot.getFromDef("CAN")
print(can)