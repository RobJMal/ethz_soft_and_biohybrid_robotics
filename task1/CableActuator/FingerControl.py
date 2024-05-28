# Required import for SOFA within python
import Sofa
import Sofa.Core
import Sofa.Simulation
import Sofa.Gui

# -*- coding: utf-8 -*-
import os

path = os.path.dirname(os.path.abspath(__file__)) + '/mesh/'

# For the controller 
import numpy as np 


def main():
        # Call the SOFA function to create the root node
        root = Sofa.Core.Node("root")

        # Call the createScene function, as runSofa does
        createScene(root)

        # Once defined, initialization of the scene graph
        Sofa.Simulation.init(root)

        # Launch the GUI (qt or qglviewer)
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 800)

        # Initialization of the scene will be done here
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
        print("Closed GUI")


def createScene(rootNode):
    rootNode.addObject('RequiredPlugin',
                       pluginName='SoftRobots SoftRobots.Inverse')
    rootNode.addObject('VisualStyle',
                       displayFlags='showVisualModels hideBehaviorModels showCollisionModels '
                                    'hideBoundingCollisionModels hideForceFields showInteractionForceFields '
                                    'hideWireframe')

    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('QPInverseProblemSolver', printLog=False)

    rootNode.gravity = [0, -9180, 0]

    ##########################################
    # FEM Model                              #
    ##########################################
    finger = rootNode.addChild('finger')
    finger.addObject('EulerImplicitSolver', firstOrder=True, rayleighMass=0.1, rayleighStiffness=0.1)
    finger.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixMat3x3d")
    finger.addObject('MeshVTKLoader', name='loader', filename=path + 'finger.vtk')
    finger.addObject('MeshTopology', src='@loader', name='container')
    finger.addObject('MechanicalObject')
    finger.addObject('UniformMass', totalMass=0.075)
    finger.addObject('TetrahedronFEMForceField', poissonRatio=0.3, youngModulus=600)
    finger.addObject('BoxROI', name='ROI1', box=[-15, 0, 0, 5, 10, 15], drawBoxes=True)
    finger.addObject('RestShapeSpringsForceField', points='@ROI1.indices', stiffness=1e12)
    finger.addObject('LinearSolverConstraintCorrection')

    ##########################################
    # Visualization                          #
    ##########################################
    fingerVisu = finger.addChild('visu')
    fingerVisu.addObject('MeshSTLLoader', filename=path + "finger.stl", name="loader")
    fingerVisu.addObject('OglModel', src="@loader", template='Vec3', color=[0.0, 0.7, 0.7, 1])
    fingerVisu.addObject('BarycentricMapping')

    ##########################################
    # Cable                                  #
    ##########################################
    cable = finger.addChild('cable')
    cable.addObject('MechanicalObject',
                    position=[
                        [-17.5, 12.5, 2.5],
                        [-32.5, 12.5, 2.5],
                        [-47.5, 12.5, 2.5],
                        [-62.5, 12.5, 2.5],
                        [-77.5, 12.5, 2.5],

                        [-83.5, 12.5, 4.5],
                        [-85.5, 12.5, 6.5],
                        [-85.5, 12.5, 8.5],
                        [-83.5, 12.5, 10.5],

                        [-77.5, 12.5, 12.5],
                        [-62.5, 12.5, 12.5],
                        [-47.5, 12.5, 12.5],
                        [-32.5, 12.5, 12.5],
                        [-17.5, 12.5, 12.5]
                    ])

    # Set a maximum displacement for your cable
    cable.addObject('CableActuator', name="aCable",
                    indices=list(range(0, 14)),
                    pullPoint=[0.0, 12.5, 2.5],
                    maxPositiveDisp=40,
                    maxDispVariation=0.5,
                    minForce=0)

    cable.addObject('BarycentricMapping')

    ##########################################
    # Effector goal for interactive control  #
    ##########################################
    goal = rootNode.addChild('goal')
    goal.addObject('EulerImplicitSolver', firstOrder=True)
    goal.addObject('CGLinearSolver', iterations=100, tolerance=1e-5, threshold=1e-5)
    goal.addObject('MechanicalObject', name='goalMO',
                   position=[-103, 7, 7])
    goal.addObject('SphereCollisionModel', radius=5)
    goal.addObject('UncoupledConstraintCorrection')

    ##########################################
    # Effector                               #
    ##########################################
    effector = finger.addChild('fingertip')
    effector.addObject('MechanicalObject', position=([-103, 7, 7]))
    effector.addObject('PositionEffector', template='Vec3',
                       indices=0,
                       effectorGoal="@../../goal/goalMO.position")
    effector.addObject('BarycentricMapping', mapForces=False, mapMasses=False)

    goal.addObject(GoalPositionController(goal.goalMO))

    return rootNode


class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def as_list(self):
        return [self.x, self.y, self.z]
    
    def set(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"


class GoalPositionController(Sofa.Core.Controller):
    def __init__(self, goal_obj, min_range=0.0, max_range=14.0, step=0.01):
        Sofa.Core.Controller.__init__(self)
        self.goal = goal_obj

        self.x_min_range, self.x_max_range = -93.0, -104.0
        self.y_min_range, self.y_max_range = -3.5, 17.5

        self.init_position = Vector3(x = goal_obj.position.value[0][0],
                                     y = goal_obj.position.value[0][1],
                                     z = goal_obj.position.value[0][2])
        
        self.goal_controller_position = Vector3(x = self.init_position.x, 
                                                y = self.init_position.y, 
                                                z = self.init_position.z)
        self.direction = 1
        self.step = step

        self.new_x = self.init_position.x
        self.new_y = self.init_position.y
        self.new_z = self.init_position.z

        self.theta = 0.0 
        self.theta_min_range, self.theta_max_range = (-np.pi), (np.pi)
        self.ang_vel = 0.01

        self.x_amplitude = 10.0
        self.y_amplitude = 10.0

    def onEvent(self, event):
        self.theta += self.ang_vel
        self.new_x = (self.init_position.x + self.x_amplitude) + -1 * self.x_amplitude * np.cos(self.theta)
        # self.new_x = (self.init_position.x) + (-103) * np.cos(self.theta + (-1*np.pi/2))
        # self.new_x += self.step * self.direction
        
        self.new_y = self.init_position.y + (self.y_amplitude * np.sin(self.theta))
        # self.new_y += self.step * self.direction

        self.new_z = self.init_position.z

        if self.theta >= self.theta_max_range or self.theta <= self.theta_min_range:
            self.ang_vel *= -1

        # if abs(self.new_x) >= abs(self.x_max_range) or abs(self.new_x) <= abs(self.x_min_range):
        #     self.direction *= -1
        #     self.new_x += self.step * self.direction

        # if self.new_y >= self.y_max_range or self.new_y <= self.y_min_range:
        #     self.direction *= -1
        #     self.new_y += self.step * self.direction

        print(f"theta: {self.theta}")
        print(f"x: {self.new_x}, y: {self.new_y}")

        self.goal_controller_position.set(self.new_x, self.new_y, self.new_z)
        self.goal.position.value = [self.goal_controller_position.as_list()]

# Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()