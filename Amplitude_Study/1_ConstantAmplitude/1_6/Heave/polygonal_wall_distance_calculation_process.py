
import numpy as np

import KratosMultiphysics as KM
import KratosMultiphysics.RANSApplication as KratosRANS
#import KratosMultiphysics.RANSApplication.RansFormulationProcess as RansFormulationProcess


def Factory(params, Model):
    if(type(params) != KM.Parameters):
        raise Exception(
            'expected input shall be a Parameters object, encapsulating a json string')
    return PolygonalWallDistanceCalculationProcess(Model, params['Parameters'])


class PolygonalWallDistanceCalculationProcess(KM.RANSApplication.RansFormulationProcess):
    '''
    Computes the perpendicluar distance between a node and a structure
    that has a polygonal shape. This means it only works in 2D
    '''

    def __init__(self, Model, params):
        KM.RANSApplication.RansFormulationProcess.__init__(self)

        # Detect 'End' as a tag and replace it by a large number
        if(params.Has('interval')):
            if(params['interval'][1].IsString()):
                if(params['interval'][1].GetString() == 'End'):
                    params['interval'][1].SetDouble(1e30)
                else:
                    raise Exception('The second value of interval can be \'End\' or a number, interval currently:' +
                                    params['interval'].PrettyPrintJsonString())

        # Compare and fill with default values the input parameters
        default_settings = KM.Parameters("""
            {
                "fluid_model_part_name" : "",
                "structure_model_part_name": "",
                "polygon_corner_ids" : [],
                "polygon_sizes" : [],
                "calculate_each_time_step" : true
            }
            """)
        params.ValidateAndAssignDefaults(default_settings)

        # Save fluid model parts (to calculate the distance in its nodes)
        # and structure model part (to seach for the nodes that generate the walls)
        self.model_part_name = params['fluid_model_part_name'].GetString()
        self.fluid_model_part = Model[self.model_part_name]
        self.structure_model_part_name = params['structure_model_part_name'].GetString()
        self.structure_model_part = Model[self.structure_model_part_name]

        # If this is true, wall distances will be calculated each time step
        # If not, only at the start of the simulation
        self.calculate_each_time_step = params["calculate_each_time_step"].GetBool()

        # Save the IDs of the nodes that are the corners of the polygons
        node_id_list = [int(v) for v in params['polygon_corner_ids'].GetVector()]

        # Read how many sides has each polygon
        polygon_sizes = [int(v) for v in params['polygon_sizes'].GetVector()]

        # If there are no specified polygon sizes,
        # take only one polygon as a default, so its size is the total number of nodes
        if len(polygon_sizes) == 0:
            polygon_sizes.append(len(node_id_list))

        # Check that the total of all the polygon sizes is equal to the number of nodes given
        if np.sum(polygon_sizes) != len(node_id_list):
            raise Exception("The sizes of the polygons given does not match the number of nodes specified")

        # Form the edges of the polygons (specifying starting and end node IDs)
        self.structure_lines = self._FormStructureLines(node_id_list, polygon_sizes)


    def _FormStructureLines(self, ids, sizes):

        # We will have as many lines as nodes, or as the sum of the sizes of all polygons
        n_nodes = int(np.sum(sizes))
        lines = np.zeros((n_nodes, 2),dtype=int)

        # Each node serves as a start to one line
        lines[:,0] = ids

        # Prepare the lines of each polygon
        first_node_index = 0
        for poly_size in sizes:

            # Prepare each line of the polygon
            for corner_id in range(poly_size):

                # If it is not the last line of the polygon,
                # the end of the line is the same as the start of the next one
                if corner_id+1 != poly_size:
                    lines[first_node_index+corner_id,1] = lines[first_node_index+corner_id+1,0]

                # If it is the last line,
                # the end of the line is the start of the first line of the polygon
                else:
                    lines[first_node_index+corner_id,1] = lines[first_node_index,0]

            # Increase the first node id so the next polygon has a different starting point
            first_node_index += poly_size

        return lines


    def ExecuteInitialize(self):

        # If only want to calculate the distances once, calculate it here
        if not self.calculate_each_time_step:
            self._ComputeWallDistances()


    def ExecuteInitializeSolutionStep(self):

        # If we want to calculate the distances every time step, do it here
        if self.calculate_each_time_step:
            self._ComputeWallDistances()


    def _ComputeWallDistances(self):

        # Mark start of the calculation
        print("PolygonalWallDistanceCalculationProcess: Calculating wall distances...")

        KratosRANS.RansCalculationUtilities.CalculateWallDistances(self.fluid_model_part, self.structure_lines)

        # Mark end of the calculation
        print("PolygonalWallDistanceCalculationProcess: Wall distance calculation completed")