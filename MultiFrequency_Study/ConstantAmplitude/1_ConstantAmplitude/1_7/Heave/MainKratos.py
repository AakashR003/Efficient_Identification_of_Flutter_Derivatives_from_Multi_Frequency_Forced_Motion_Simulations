import os

import KratosMultiphysics
import KratosMultiphysics.StatisticsApplication
from KratosMultiphysics.RANSApplication.rans_analysis import RANSAnalysis


if __name__ == "__main__":
    
    # Change working directory so all results are written from the base folder
    cwd = os.path.dirname(__file__)
    os.chdir(cwd)

    # Read the already prepared project parameters
    with open("ProjectParameters_Custom.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    '''
    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    global_model = KratosMultiphysics.Model()
    simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters)
    simulation.Run()
    '''
    
    # Run the simulation
    model = KratosMultiphysics.Model()
    simulation = RANSAnalysis(model, parameters)
    simulation.Run()