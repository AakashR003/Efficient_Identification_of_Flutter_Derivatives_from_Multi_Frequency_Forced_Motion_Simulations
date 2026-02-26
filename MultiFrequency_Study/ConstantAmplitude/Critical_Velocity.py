
import numpy as np
from scipy.linalg import eig


#Structural Properties

mass = 10 # kg/m
MoI = 5
vertical_frequency = 1.0 # Hz
rotational_frequency = 0.5 # Hz
vertical_damping_ratio = 0.02
rotational_damping_ratio = 0.02

#Aerodynamic Properties

breadth = 0.366 # m
u = 10.0 # m/s
reduced_u = 7 # m/s
rho = 1.225 # kg/m^3

#Aerodynamic Coefficients
A1 = 0.1
A2 = 0.05
A3 = 0.01
A4 = 0.005

H1 = 0.2
H2 = 0.1
H3 = 0.02
H4 = 0.01

#Calcuated Structural Properties
vertical_stiffness = (2*np.pi*vertical_frequency)**2 * mass
rotational_stiffness = (2*np.pi*rotational_frequency)**2 * MoI
vertical_damping_coefficient = 2 * mass * (2*np.pi*vertical_frequency) *  vertical_damping_ratio
rotational_damping_coefficient = 2 * MoI * (2*np.pi*rotational_frequency) *  rotational_damping_ratio

#Calculated Aerodynamic Properties
omega = u / reduced_u * 2 * np.pi
reduced_frequency = breadth * omega  / u


def Structural_Properties_Matrix():

    M = [[mass, 0],
         [0, MoI]]
    
    K = [[vertical_stiffness, 0],
         [0, rotational_stiffness]]
    
    C = [[vertical_damping_ratio, 0],
         [0, rotational_damping_ratio]]
    
    return M, K, C


def Aerodynamic_Properties_Matrix(H1=H1, H2=H2, H3=H3, H4=H4, A1=A1, A2=A2, A3=A3, A4=A4, reduced_frequency=reduced_frequency):

    
    Lift_Force = 0.5 * rho * u**2 * breadth
    Moment_Force = 0.5 * rho * u**2 * breadth * breadth

    # Aerodynamic stiffness matrix
    C = [[Lift_Force * reduced_frequency * H1 * omega / u, Lift_Force * reduced_frequency * H2 * omega * breadth / u],
        [Moment_Force * reduced_frequency * A1 * omega / u, Moment_Force * reduced_frequency * A2 * omega * breadth/ u]]

    # Aerodynamic damping matrix
    K = [[Lift_Force * reduced_frequency**2 * H3, Lift_Force * reduced_frequency**2 * H4 / breadth],
        [Moment_Force * reduced_frequency**2 * A3, Moment_Force * reduced_frequency**2 * A4 / breadth]]

    return C, K 

def Total_Properties_Matrix(H1=H1, H2=H2, H3=H3, H4=H4, A1=A1, A2=A2, A3=A3, A4=A4, reduced_frequency=reduced_frequency):

    M_struct, K_struct, C_struct = Structural_Properties_Matrix()
    C_aero, K_aero = Aerodynamic_Properties_Matrix(H1=H1, H2=H2, H3=H3, H4=H4, A1=A1, A2=A2, A3=A3, A4=A4, reduced_frequency=reduced_frequency)

    M_total = M_struct

    C_total = [[C_struct[0][0] - C_aero[0][0], C_struct[0][1] - C_aero[0][1]],
               [C_struct[1][0] - C_aero[1][0], C_struct[1][1] - C_aero[1][1]]]
    
    K_total = [[K_struct[0][0] - K_aero[0][0], K_struct[0][1] - K_aero[0][1]],
               [K_struct[1][0] - K_aero[1][0], K_struct[1][1] - K_aero[1][1]]]

    return M_total, C_total, K_total

def Fomrulate_Qudratic_Eigenvalue_Problem():

    M, C, K = Total_Properties_Matrix(H1=H1, H2=H2, H3=H3, H4=H4, A1=A1, A2=A2, A3=A3, A4=A4, reduced_frequency=reduced_frequency)
    quadratic_equation = M*omega**2 + C*omega + K

    return quadratic_equation

def Linearzed_General_Eigen_Value_Problem(H1=H1, H2=H2, H3=H3, H4=H4, A1=A1, A2=A2, A3=A3, A4=A4, reduced_frequency=reduced_frequency):

    M, C, K = Total_Properties_Matrix(H1=H1, H2=H2, H3=H3, H4=H4, A1=A1, A2=A2, A3=A3, A4=A4, reduced_frequency=reduced_frequency)

    A = [[0, K],
         [K, C]]
    
    B = [[K, 0],
         [0, -M]]
    
    return A, B

def Solve_Eigen_Value():

    A, B = Linearzed_General_Eigen_Value_Problem(H1=H1, H2=H2, H3=H3, H4=H4, A1=A1, A2=A2, A3=A3, A4=A4, reduced_frequency=reduced_frequency)
    eigenvalues, eigenvectors = eig(A, B)

    return eigenvalues, eigenvectors

def omega():

    eigenvalues, eigenvectors = Solve_Eigen_Value()
    omega = eigenvalues[1]/eigenvalues[0]
    return omega
    