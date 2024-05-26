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
    # First Finger                           #
    ##########################################
    finger1 = rootNode.addChild('finger1')
    finger1.addObject('EulerImplicitSolver', firstOrder=True, rayleighMass=0.1, rayleighStiffness=0.1)
    finger1.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixMat3x3d")
    finger1.addObject('MeshVTKLoader', name='loader', filename=path + 'finger.vtk')
    finger1.addObject('MeshTopology', src='@loader', name='container')
    finger1.addObject('MechanicalObject')
    finger1.addObject('UniformMass', totalMass=0.075)
    finger1.addObject('TetrahedronFEMForceField', poissonRatio=0.3, youngModulus=600)
    finger1.addObject('BoxROI', name='ROI1', box=[-15, 0, 0, 5, 10, 15], drawBoxes=True)
    finger1.addObject('RestShapeSpringsForceField', points='@ROI1.indices', stiffness=1e12)
    finger1.addObject('LinearSolverConstraintCorrection')

    finger1Visu = finger1.addChild('visu')
    finger1Visu.addObject('MeshSTLLoader', filename=path + "finger.stl", name="loader")
    finger1Visu.addObject('OglModel', src="@loader", template='Vec3', color=[0.0, 0.7, 0.7, 1])
    finger1Visu.addObject('BarycentricMapping')

    cable1 = finger1.addChild('cable')
    cable1.addObject('MechanicalObject', 
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
    cable1.addObject('CableActuator', name="aCable", indices=list(range(0, 14)), pullPoint=[0.0, 12.5, 2.5], maxPositiveDisp=40, maxDispVariation=0.5, minForce=0)
    cable1.addObject('BarycentricMapping')

    goal1 = rootNode.addChild('goal1')
    goal1.addObject('EulerImplicitSolver', firstOrder=True)
    goal1.addObject('CGLinearSolver', iterations=100, tolerance=1e-5, threshold=1e-5)
    goal1.addObject('MechanicalObject', name='goalMO', position=[-120, 7, 7])
    goal1.addObject('SphereCollisionModel', radius=5)
    goal1.addObject('UncoupledConstraintCorrection')

    effector1 = finger1.addChild('fingertip')
    effector1.addObject('MechanicalObject', position=([-103, 7, 7]))
    effector1.addObject('PositionEffector', template='Vec3', indices=0, effectorGoal="@../../goal1/goalMO.position")
    effector1.addObject('BarycentricMapping', mapForces=False, mapMasses=False)

    return rootNode


# Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()