import sys
import math

import KratosMultiphysics
from KratosMultiphysics.time_based_ascii_file_writer_utility import TimeBasedAsciiFileWriterUtility
import KratosMultiphysics.MeshMovingApplication as MeshMovingApplication


def Factory(params, Model):
    if(type(params) != KratosMultiphysics.Parameters):
        raise Exception(
            'expected input shall be a Parameters object, encapsulating a json string')
    return ForcedMotionWithDistortionProcess(Model, params["Parameters"])


class ForcedMotionWithDistortionProcess(KratosMultiphysics.Process):
    '''
    Imposes one or several super-imposed sinusoidal motions
    to a single box deck cross-section.
    For each sinusoidal motion, the DoF (w, theta) and 
    the frequency can be configured.
    
    The process outputs the resultant forces in the box
    as well as the imposed motion.
    '''

    def __init__(self, Model, params):
        KratosMultiphysics.Process.__init__(self)

        default_settings = KratosMultiphysics.Parameters("""
            {
                "model_part_name_1"      : "",
                "interval"               : [0.0, 1e30],
                "rampup_time"            : 0.0,
                "reference_point_1"      : [0.0,0.0,0.0],
                "imposed_motion":{
                    "dofs"               : [],
                    "frequencies"        : [],
                    "ureds"              : [],
                    "amplitudes"         : [],
                    "U"                  : -1,
                    "B"                  : -1
                },
                "write_output_files"     : true,
                "output_format"          : ".8f",
                "output_file_settings"   : {}
            }
            """)

        # Detect 'End' as a tag and replace it by a large number
        if(params.Has('interval')):
            if(params['interval'][1].IsString()):
                if(params['interval'][1].GetString() == 'End'):
                    params['interval'][1].SetDouble(1e30)
                else:
                    raise Exception('The second value of interval can be \'End\' or a number, interval currently:' +
                                    params['interval'].PrettyPrintJsonString())

        # Fill parameters with default values
        params.ValidateAndAssignDefaults(default_settings)
        params["imposed_motion"].ValidateAndAssignDefaults(default_settings["imposed_motion"])

        # Save parameters in class variables
        self.model_part_names = {}
        self.model_parts = None
        self.reference_points = {}
        self.model_part_names = params["model_part_name_1"].GetString()
        self.model_parts = Model[self.model_part_names]
        self.reference_points = params["reference_point_1"].GetVector()
        self.interval = params["interval"].GetVector()
        self.rampup_time = params['rampup_time'].GetDouble()
        self.write_output_files = params['write_output_files'].GetBool()
        self.format = params["output_format"].GetString()

        # Motion parameters
        self.dofs = params["imposed_motion"]["dofs"].GetStringArray()
        self.frequencies = list(params["imposed_motion"]["frequencies"].GetVector())
        self.amplitudes = list(params["imposed_motion"]["amplitudes"].GetVector())

        # Check that all the DOFs are among the available ones
        available_dofs = ["w1", "theta1", "w", "theta"]
        for dof in self.dofs:
            if dof not in available_dofs:
                msg = 'The DOF "' + dof + '" is not among the available ones: '
                msg += str(available_dofs)[1:-1]
                raise Exception(msg)
        
        # If reduced wind speeds are provided, calculate the frequencies from them instead
        ureds = list(params["imposed_motion"]["ureds"].GetVector())
        if len(ureds) != 0:

            # Check that 'U' and 'B' were given
            U = params["imposed_motion"]["U"].GetDouble()
            if U == -1:
                msg = "The wind speed 'U' needs to be specified if "
                msg += "the reduced wind speeds 'ureds' are used as the input."
                raise Exception(msg)
            B = params["imposed_motion"]["B"].GetDouble()
            if B == -1:
                msg = "The box width 'B' needs to be specified if "
                msg += "the reduced wind speeds 'ureds' are used as the input."
                raise Exception(msg)

            # Calculate the frequencies from the reduced wind speeds
            self.frequencies = [U/ured/B for ured in ureds]

        # Check that we have the same DOFs and frequencies
        if len(self.dofs) != len(self.frequencies):
            msg = "The number of frequencies provided needs to match "
            msg += "the number of DOFs given. "
            raise Exception(msg)

        # Calculate amplitudes automatically if necessary
        if len(self.amplitudes) == 0:

            # Check that 'B' was given
            B = params["imposed_motion"]["B"].GetDouble()
            if B == -1:
                msg = "The box width 'B' needs to be specified if "
                msg += "the amplitudes are not manually provided."
                raise Exception(msg)

            # Setting maximum amplitudes depending on the motion type (angular or linear)
            max_amplitudes = {
                "linear_local" : B*0.05,
                "angular" : 2*math.pi/180
            }
            for dof in self.dofs:
                if dof in ["w", "w1"]:
                    self.amplitudes.append(max_amplitudes["linear_local"]/len(self.dofs))
                elif dof in ["theta1", "theta"]:
                    self.amplitudes.append(max_amplitudes["angular"]/len(self.dofs))

        # Check that we have the same DOFs and amplitudes (this is only
        # necessary if the amplitudes were not generated automatically)
        elif len(self.dofs) != len(self.amplitudes):
            msg = "The number of amplitudes provided needs to match "
            msg += "the number of DOFs given. "
            raise Exception(msg)  
        
        # Configure output files
        if (self.write_output_files):

            # Read the output-specific parameters
            file_handler_params = KratosMultiphysics.Parameters(params["output_file_settings"])

            # Overwrite the file name if it was sepecified
            # Create an empty field if there was no file name specified
            if file_handler_params.Has("file_name"):
                output_file_name = file_handler_params["file_name"].GetString()
            else:
                file_handler_params.AddEmptyValue("file_name")
                # A default file name
                output_file_name = "forced_motion_output"

            # Both files will be stored in this dictionary
            self.output_files = {}

            # Create files for the single box
            self.output_files = {}
                
            # Only write output files if this is rank 0
            if (self.model_parts.GetCommunicator().MyPID() == 0):

                # Create different files for the forces and motion
                for case in ['motion', 'forces']:

                    # Clone the parameters so we can overwrite the file name
                    case_file_handler_params = file_handler_params.Clone()
                    
                    # Add flags to the file names to differenciate which box and output type it is
                    case_file_handler_params["file_name"].SetString(output_file_name + '_' + case + '.dat')

                    # Create file
                    file_header = self._GetFileHeader(case)
                    self.output_files[case] = TimeBasedAsciiFileWriterUtility(
                        self.model_parts,
                        case_file_handler_params,
                        file_header
                    ).file


    def ExecuteInitializeSolutionStep(self):
        
        # Update imposed motion and reference points
        self._UpdateSolutionStepIncrement()
        
        # C++ command for the rotation of the mesh
        MeshMovingApplication.MoveModelPart(
            self.model_parts,
            [0, 0, 1],                                        # rotation axis
            self.motion["theta"],                         # rotation angle
            self.reference_points,                       # one point of the rotation axis
            [self.motion["v"], self.motion["w"], 0])  # translation (after the rotation)


    def _UpdateSolutionStepIncrement(self):
        # Updates the heave and pitch values for the box in the current time step
        # and also adjusts the current reference points for the calculation of the forces

        # Get current time
        time = self.model_parts.ProcessInfo[KratosMultiphysics.TIME]

        # Restart the motion increments
        self.motion = {
            "w1" : 0,
            "v1" : 0,
            "theta1" : 0
        }

        # Add motion only if it is inside the interval
        if time >= self.interval[0] and time < self.interval[1]:

            # Add individually each of the super-imposed motions
            for dof, freq, amp in zip(self.dofs, self.frequencies, self.amplitudes):
                
                # Contribution of this single-frequency component of the total motion
                motion_contribution = amp*math.sin(2*math.pi*freq*time)

                # Correct to consider rampup if necessary
                if time < self.rampup_time:
                    motion_contribution *= 0.5 * (math.cos(math.pi*(1-(time/self.rampup_time))) + 1)

                # Depending on the excited DOF, we will add the motion contribution
                # only to specific displacements ("w1" or "theta1")
                if dof in ["w1", "theta1"]:
                    self.motion[dof] += motion_contribution
                elif dof == "w":
                    self.motion["w1"] += motion_contribution
                elif dof == "theta":
                    self.motion["theta1"] += motion_contribution
            
        # Note updated reference center of the box for the calculation of the moments
        self.current_reference_points = {}
        self.current_reference_points = [
            self.motion["v"],
            self.motion["w"],
            0
        ]


    def ExecuteFinalizeSolutionStep(self):

        # Get current time
        time = self.model_parts.ProcessInfo[KratosMultiphysics.TIME]

        # Write output only if the option was selected
        if self.write_output_files:

            # Calculate resultant forces and moments acting on the box
            force, moment = self._EvaluateGlobalForces()

            # Only write output files if this is rank 0
            if (self.model_parts.GetCommunicator().MyPID() == 0):

                # Format line to be printed in the file
                force_output = force + moment
                motion_output = [self.motion["v"], self.motion["w"], self.motion["theta"]]
                force_output = [format(val, self.format) for val in force_output]
                force_output.insert(0, str(time))
                motion_output = [format(val, self.format) for val in motion_output]
                motion_output.insert(0, str(time))

                # Write time step data
                self.output_files['forces'].write(' '.join(force_output) + '\n')
                self.output_files['motion'].write(' '.join(motion_output) + '\n')


    def _EvaluateGlobalForces(self):
        # Calculates the resultant forces and moments acting on the box

        # C++ function to extract flow-attached forces: in x-y-z coordinate system
        reaction_force, reaction_moment = KratosMultiphysics.ForceAndTorqueUtils.ComputeEquivalentForceAndTorque(
            self.model_parts,
            self.current_reference_points,
            KratosMultiphysics.REACTION,
            KratosMultiphysics.REACTION_MOMENT
        )

        # Invert sign to go from reaction to resultant
        force = list(-1*reaction_force)
        moment = list(-1*reaction_moment)

        return force, moment


    def _GetFileHeader(self, case):
        # Generates the headers of the ascii file depending on which output it has

        if case == "motion":
            header = '# Motion for model part ' + self.model_part_names + '\n'
            header += '# Time v w theta\n'
        elif case == "forces":
            header = '# Forces for model part ' + self.model_part_names + '\n'
            header += '# Time Fx Fy Fz Mx My Mz\n'

        return header