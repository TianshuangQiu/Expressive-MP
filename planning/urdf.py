from urdfpy import URDF

robot = URDF.load('/home/akita/autolab/ur5/ur5_description/urdf/ur5_joint_limited_robot.urdf')
#robot = URDF.load('ur5.urdf')
for link in robot.links:    
    print(link.name)

fk = robot.link_fk()
print(fk[robot.links[2]])

fk = robot.visual_trimesh_fk()
print(type(list(fk.keys())[0]))
robot.show()