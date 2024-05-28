import numpy as np
from scipy.spatial.transform import Rotation as R

import Sofa.Core
import Sofa.constants.Key as Key
import Sofa.Simulation

from stlib3.scene import MainHeader, ContactHeader
from stlib3.physics.rigid import Floor, Cube

from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox
from softrobots.actuators import PullingCable
from stlib3.physics.collision import CollisionMesh
from splib3.loaders import loadPointListFromFile


def transform_points(points, rotation, translation):
    #TODO: Write a function to transform points using the rotation and translation parameters.
    transformed_points = points  # Replace this line
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
        
        # Here's how you can create a single finger
        fingers.append(createFinger("fingerName",
                    parentNode=parentNode,
                    rotation=[0.0, 0.0, 0.0],
                    translation=[0.0, 0.0, 0.0],
                    boxCoords=[[-5.0, 0.0, -5.0], [10.0, 10.0, 20.0]],
                    pullPointLocation=[0.0, 0.0, 0.0]))
        
        print(f"{fingerName} created")


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

    # TODO: Create a sphere to grasp
