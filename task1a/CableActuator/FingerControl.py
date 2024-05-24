# Required import for SOFA within python
import Sofa


def main():
        # Call the SOFA function to create the root node
        root = Sofa.Core.Node("root")

        # Call the createScene function, as runSofa does
        createScene(root)

        # Once defined, initialization of the scene graph
        Sofa.Simulation.init(root)

        # Run as many simulation steps (here 10 steps are computed)
        for iteration in range(10):
                Sofa.Simulation.animate(root, root.dt.value)


# Same createScene function as in the previous case
def createScene(rootNode):
        #Doesn't do anything yet
        return rootNode


# Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()