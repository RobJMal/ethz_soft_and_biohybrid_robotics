import numpy as np
from scipy.spatial.transform import Rotation as R

import Sofa.Core
import Sofa.constants.Key as Key
import Sofa.Simulation

# Helps program find stlib3 packages 
import sys
stlib3_path = '/home/robjmal/SOFA/v23.12.01/plugins/STLIB/lib/python3/site-packages'
sys.path.append(stlib3_path)
softrobots_path = '/home/robjmal/SOFA/v23.12.01/plugins/SoftRobots/lib/python3/site-packages'
sys.path.append(softrobots_path)

from stlib3.scene import MainHeader, ContactHeader
from stlib3.physics.rigid import Floor, Cube

from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox
from softrobots.actuators import PullingCable
from stlib3.physics.collision import CollisionMesh
from splib3.loaders import loadPointListFromFile


# Helper functions for transforming points
def rotation_matrix_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_matrix_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_matrix_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
        [0, 0, 1]
    ])
    
def calculate_full_rotation(rx, ry, rz):
    return np.dot(np.dot(rotation_matrix_x(rx), rotation_matrix_y(ry)), rotation_matrix_z(rz))

def transform_points(points, rotation, translation):
    #TODO: Write a function to transform points using the rotation and translation parameters.
    '''
    Transform points using the rotation and translation parameters. 
    '''
    rotation_angle_x, rotation_angle_y, rotation_angle_z = rotation[0], rotation[1], rotation[2]
    translation_x, translation_y, translation_z = translation[0], translation[1], translation[2]

    rotation_matrix = calculate_full_rotation(rotation_angle_x, rotation_angle_y, rotation_angle_z)

    transformed_points = []

    for _, point in enumerate(points):
        point_nparray = np.array(point)

        # Apply transformation 
        transformed_point = np.dot(rotation_matrix, point) + np.array(translation)
        transformed_points.append(transformed_point.to_list())

    return transformed_points


def createFinger(name,
                 parentNode,
                 rotation,
                 translation,
                 boxCoords,
                 pullPointLocation):

    fingerObj = parentNode.addChild(name)
    elasticMeshObj = ElasticMaterialObject(fingerObj,
                    volumeMeshFileName="mesh/finger.vtk",
                    poissonRatio=0.3,
                    youngModulus=18000,
                    totalMass=0.5,
                    surfaceColor=[0.0, 1.0, 0.5, 1.0],
                    surfaceMeshFileName="mesh/finger.stl",
                    rotation=rotation,
                    translation=translation)
    fingerObj.addChild(elasticMeshObj)
    print(f"{name} :: ElasticMaterialObject created")

    FixedBox(elasticMeshObj, atPositions=[*boxCoords[0], *boxCoords[1]], doVisualization=True)
    cableObj = PullingCable(elasticMeshObj,
                         "PullingCable",
                         pullPointLocation=pullPointLocation,
                         rotation=rotation,
                         translation=translation,
                         cableGeometry=loadPointListFromFile("cablePoints.json"))
    cableObj.addObject('BarycentricMapping')
    print(f"{name} :: cableObj created")

    CollisionMesh(elasticMeshObj, name="CollisionMesh",
                    surfaceMeshFileName="mesh/finger.stl",
                    rotation=rotation, translation=translation,
                    collisionGroup=[1, 2])

    elasticMeshObj.addObject(FingerController(cable=cableObj, name=name))


class FingerController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.cable = kwargs["cable"]
        self.name = f'{kwargs["name"]}Controller'
        print(f"{self.name} created")

    def onKeypressedEvent(self, e):
        """Handle key presses to open and close the gripper."""
        # Example of how to handle key presses to open and close the fingers
        # You will instead open and close fingers autonomously to grasp the object within
        # the state machine you will implement in the GripperController

        print("Key pressed: ", e["key"])
        displacement = self.cable.CableConstraint.value[0]
        if ord(e["key"]) == ord(']'):  # Press ] to close the gripper
            displacement += 1.
        elif ord(e["key"]) == ord('['):  # Press [ to open the gripper
            displacement -= 1.
            if displacement < 0:
                displacement = 0
        self.cable.CableConstraint.value = [displacement]


class GripperController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.fingers = kwargs["fingers"]
        self.name = "GripperController"
        print(f"{self.name} created")

    def onAnimateBeginEvent(self, eventType):
        # TODO: Write a simple state machine to control the gripper
        # state machine to move gripper grasp into the right position
        # grasp the object, and then move up
        pass


def createGripper(name, parentNode):
    fingers = []

    for i in range(3):
        fingerName = f"{name}-finger{i}"
        # Create a finger here
        finger_rotation_y = i*120.0

        # Here's how you can create a single finger
        fingers.append(createFinger("fingerName",
                    parentNode=parentNode,
                    rotation=[0.0, finger_rotation_y, 0.0],
                    translation=[0.0, 50.0, 0.0],
                    boxCoords=[[-10.0, 0.0, -10.0], [20.0, 100.0, 40.0]],
                    pullPointLocation=[0.0, 50.0, 0.0]))
        
        print(f"{fingerName} created")


def createSphere(name, parentNode):
    # Mechanical properties of sphere to incorporate gravity
    totalMass = 1.0
    volume = 1.0
    inertiaMatrix=[1., 0., 0., 0., 1., 0., 0., 0., 1.]

    # Creating sphere to grasp that falls due to gravity   
    sphere = parentNode.addChild(name)
    sphere_radius = 40
    sphere.addObject('EulerImplicitSolver', name='odesolver')
    sphere.addObject('CGLinearSolver', name='Solver', iterations=25, tolerance=1e-05, threshold=1e-05)
    sphere.addObject('MechanicalObject', name="mstate", template="Rigid3", translation2=[0., -20., 0.], rotation2=[0., 0., 0.], showObjectScale=sphere_radius)
    sphere.addObject('UniformMass', name="mass", vertexMass=[totalMass, volume, inertiaMatrix[:]])
    sphere.addObject('UncoupledConstraintCorrection')

    # Visualization subnode for the sphere
    sphereVisu = sphere.addChild("VisualModel")
    sphereVisu.loader = sphereVisu.addObject('MeshOBJLoader', name="loader", filename="mesh/ball.obj")
    sphereVisu.addObject('OglModel', name="model", src="@loader", scale3d=[sphere_radius]*3, color=[0., 1., 0.], updateNormals=False)
    sphereVisu.addObject('RigidMapping')

    #### Collision subnode for the sphere
    collision = sphere.addChild('collision')
    collision.addObject('MeshOBJLoader', name="loader", filename="mesh/ball.obj", triangulate="true", scale=45.0)
    collision.addObject('MeshTopology', src="@loader")
    collision.addObject('MechanicalObject')
    collision.addObject('TriangleCollisionModel')
    collision.addObject('LineCollisionModel')
    collision.addObject('PointCollisionModel')
    collision.addObject('RigidMapping')


def createScene(rootNode):
    MainHeader(rootNode, gravity=[0.0, -981.0, 0.0], plugins=["SoftRobots"])
    ContactHeader(rootNode, alarmDistance=4, contactDistance=3, frictionCoef=0.08)
    rootNode.VisualStyle.displayFlags = "showBehavior showCollisionModels"

    floorObj = Floor(rootNode,
                    color=[1.0, 0.0, 0.0, 1.0],
                    translation=[0.0, -100.0, 0.0],
                    isAStaticObject=True)

    # Create a three-fingered gripper by applying the correct transformations
    gripper = createGripper("gripper", rootNode)

    # Creating sphere to grasp
    sphere = createSphere("sphere", rootNode)
