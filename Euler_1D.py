import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

def Steger_Warming_1D(space_len: float, time_len: float, x_0: float,  rho_l: float, vel_l : float, press_l : float, 
                      rho_r: float, vel_r: float, press_r: float, N_div: int, gamma: float, const_C: float):
    """Function solving Euler's equations in 1D using Steger Warming numeric flow"""

    #calculating parameters
    space_step = space_len / N_div
    #D_vol = np.array([[i*space_step, (i+1)*space_step] for i in range(-1,N_div + 1)])
    net_x_cent = np.array([(i+0.5)*space_step for i in range(-1,N_div + 1)])
    curr_t = 0
    time_list = []
    
    #defining functions needed for calculating w in next time step
    def air_vel(press: float, rho: float)-> float:
        """function that returns air velocity"""
        assert rho != 0
        assert press >= 0
        return math.sqrt(gamma * press / rho)
    
    def entalpy(press: float, rho: float, vel: float)->float:
        """function that returns entalpy"""
        return (air_vel(press, rho)**2)/(gamma - 1) + 0.5*(vel**2)
    
    def get_var_w(w_triple)->tuple:
        """funtion that returns density, velocity and pressure from vector of w"""
        rho = w_triple[0]
        vel = w_triple[1]/rho
        press = (gamma - 1)*(w_triple[2] - 0.5*rho*(vel**2))
        return rho, vel, press
    
    def calc_time_step(w_curr:np.array)-> float:
        """function that calculates length of next step"""
        lambda_list = []    #list to store absolute value of velocity at point i + air velocity at point i
        for i in range(N_div + 1):
            rho, vel, press = get_var_w(w_curr[:,i])
            lambda_list.append(abs(vel) + air_vel(press, rho))
        lambda_max = max(lambda_list)
        return const_C * space_step / lambda_max
    
    def calc_T_mat(w_triple)->np.array:
        """function that returns matrix 3x3 of eigen vectors(matrix T)"""
        rho, vel, press = get_var_w(w_triple)
        a = air_vel(press, rho)
        H = entalpy(press, rho, vel)
        return np.array([[1, 1, 1], 
                         [vel - a, vel, vel + a], 
                         [H - a * vel, 0.5*(vel**2), H + a * vel]], dtype=float)
    
    def calc_T_inv_mat(w_triple)->np.array:
        """function that returns inverse matrix of eigen vectors(matrix T^-1)"""
        rho, vel, press = get_var_w(w_triple)
        a = air_vel(press, rho)
        div_const = 1/(2*(a**2))
        return div_const * np.array([[0.5*(gamma - 1)*(vel**2) + a*vel, -a - (gamma - 1)*vel, gamma - 1],
                                    [2*(a**2) - (gamma - 1)*(vel**2), 2*(gamma - 1)*vel, -2*(gamma - 1)],
                                    [0.5*(gamma - 1)*(vel**2) - a*vel, a - (gamma -1)*vel, gamma - 1]], dtype=float)
    
    def calc_D_mat(w_triple, sign)->np.array:
        """function that returns diagonal matrix of eigen values, if sign is False then returns -1 * eigen values"""
        rho, vel, press = get_var_w(w_triple)
        a = air_vel(press, rho)
        sign_fun = max if sign else min
        return np.diag([sign_fun(vel - a,0), sign_fun(vel,0), sign_fun(vel + a,0)])
    
    def calc_S_W_flow(w_l, w_r)->np.array:
        """function that returns Steger-Warming flow"""
        # T * D^+ * T^-1
        mat_A_plus = calc_T_mat(w_l) @ calc_D_mat(w_l, True) @ calc_T_inv_mat(w_l)
        # T * D^- * T^-1
        mat_A_min = calc_T_mat(w_r) @ calc_D_mat(w_r, False) @ calc_T_inv_mat(w_r)
        # A^+ * w_l + A^- * w_r
        return mat_A_plus.dot(w_l) + mat_A_min.dot(w_r)
        

    w_curr = np.empty((3, N_div + 2), dtype = float)
    #calculate w_0 for t = 0
    for i in range(1,N_div + 1):
        if net_x_cent[i] < x_0:
            w_curr[:,i] = [rho_l, rho_l * vel_l, press_l/(gamma - 1) + 0.5*rho_l*(vel_l**2)]
        else:
            w_curr[:,i] = [rho_r, rho_r * vel_r, press_r/(gamma - 1) + 0.5*rho_r*(vel_r**2)]

    w_curr[:,0] = w_curr[:,1]
    w_curr[:,-1] = w_curr[:,-2]

    # for i in range(0, N_div+2):
    #         print(w_curr[:,i])

    w_whole = [np.copy(w_curr)] #saving values of w_0

    w_next = np.empty((3, N_div + 2), dtype = float)

    calculate = True
    #calculation of w_j^(n+1) for n+1 time
    while(calculate):
        time_step = calc_time_step(w_curr)
        #checks if we would step over time interval
        if(time_len - curr_t < 10**(-6)):
            time_step = time_len - curr_t
            curr_t = time_len
            calculate = False
        time_list.append(curr_t)
        curr_t += time_step
        for i in range(1, N_div + 1):
            # w_j^(n+1) = w_j^n - tau_n / h * (H(w_j^n, w_(j+1)^n - H(w_(j-1)^n, w_j^n))
            w_next[:,i] = w_curr[:,i] - (time_step/space_step) * (calc_S_W_flow(w_curr[:,i], w_curr[:,i+1]) - calc_S_W_flow(w_curr[:,i - 1], w_curr[:,i]))

        w_next[:,0] = w_next[:,1]
        w_next[:,-1] = w_next[:,-2]
        
        w_curr = np.copy(w_next)
        w_whole.append(np.copy(w_next))
        # for i in range(0, N_div+2):
        #     print(w_next[:,i])

    return net_x_cent, time_list, w_whole
    

def plot_Steger_Warming_1D(save, space_len: float, time_len: float, x_0: float,  rho_l: float, vel_l : float, press_l : float, 
                      rho_r: float, vel_r: float, press_r: float, N_div: int, gamma: float, const_C: float)->None:
    """Function that plots results of solving Eulers equations by Steger Warming numeric flow and comparing with sollution from matlab function"""
    #loading data from Steger_Warming numerical method
    net_x, time_l, w_whole = Steger_Warming_1D(space_len, time_len, x_0,  rho_l, vel_l, press_l, 
                      rho_r, vel_r, press_r, N_div, gamma, const_C)
    def get_var_w(w_triple)->tuple:
        """funtion that returns density, velocity and pressure from vector of w"""
        rho = w_triple[0]
        vel = w_triple[1]/rho
        press = (gamma - 1)*(w_triple[2] - 0.5*rho*(vel**2))
        return rho, vel, press
    
    print(len(time_l))
    rho = []
    vel = []
    press = []
    energy = []
    w_last = w_whole[-1]
    for i in range(N_div):
        r,v,p = get_var_w(w_last[:,i])
        rho.append(r)
        vel.append(v)
        press.append(p)
        energy.append(p/(gamma - 1) + 0.5*r*(v**2))
    data_S_W = [rho, vel, press, energy]
    
    
    #loading computed data from matlab function
    df = pd.read_excel("mat_saves/" + save, header = None, dtype=float)
    fig, axs = plt.subplots(2,2)
    # fig.subtitle("Sollution of euler's equations")
    titles = ["Density", "Velocity", "Pressure", "Energy per unit mass"]
    for i in range(2):
        for j in range(2):
            axs[i, j].plot(df.iloc[0], df.iloc[i*2 + j + 1], df.iloc[0], data_S_W[i*2+j])
            axs[i, j].set_title(titles[i*2+j])
            axs[i, j].legend(["Matlab function", "Python function"])
