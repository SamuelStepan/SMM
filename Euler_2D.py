import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
    

def Roe_HH_2D(x_inter: tuple, y_inter: tuple ,rho_ins: float, vel_x_ins: float, vel_y_ins:float, press_ins: float, rho_out: float, vel_x_out: float,
              vel_y_out:float, press_out:float, radius: float, time_len: float, cons_C: float, gamma: float, Nx_div: int, Ny_div: int):
    """Function that solves Eulers equation for 2D using Roe's numeric flow with Harten-Hyman entropy fix"""
    #calculating parameters

    step_x = (x_inter[1] - x_inter[0])/Nx_div
    step_y = (y_inter[1] - y_inter[0])/Ny_div
    net_x_cent = np.array([x_inter[0] + (i+0.5)*step_x for i in range(-1, Nx_div + 1)])
    net_y_cent = np.array([y_inter[0] + (i+0.5)*step_y for i in range(-1, Ny_div + 1)])
    curr_t = 0
    time_list = []

    #defining functions needed for calculating w in next time step
    def air_vel(press: float, rho: float)-> float:
        """function that returns air velocity"""
        assert rho != 0
        assert press >= 0
        return math.sqrt(gamma * press / rho)
    
    def entalpy(press: float, rho: float, vel_x: float, vel_y: float)->float:
        """function that returns entalpy"""
        return press * gamma / (rho * (gamma - 1)) + 0.5*(vel_x**2 + vel_y**2)
    
    def get_var_w(w_ij)->tuple:
        """funtion that returns density, velocity and pressure from vector of w"""
        rho = w_ij[0]
        vel_x = w_ij[1]/rho
        vel_y = w_ij[2]/rho
        press = (gamma - 1)*(w_ij[3] - 0.5*rho*(vel_x**2 + vel_y**2))
        return press, rho, vel_x, vel_y
    
    def calc_time_step(lambda_list:list)-> float:
        """function that calculates length of next step"""
        return min(step_x, step_y) * cons_C / (math.sqrt(2)*max(lambda_list))
    
    def calc_T_mat(w_ij, n_vec, a = None)->np.array:
        """function that returns matrix 3x3 of eigen vectors(matrix T)"""
        n_1, n_2 = n_vec[0], n_vec[1]
        press, rho, vel_x, vel_y = get_var_w(w_ij)
        if(a == None):
            a = air_vel(press, rho)
        H = entalpy(press, rho, vel_x, vel_y)
        vel_n = vel_x * n_1 + vel_y * n_2
        return np.array([[1, 1, 0, 1], 
                         [vel_x - a*n_1, vel_x, n_2, vel_x + a*n_1], 
                         [vel_y - a*n_2, vel_y, -n_1, vel_y + a*n_2],
                         [H - a*vel_n, 0.5*(vel_x**2 + vel_y**2), n_2*vel_x - n_1*vel_y, H + a*vel_n]], dtype=float)
    
    def calc_T_inv_mat(w_ij, n_vec, a = None)->np.array:
        """function that returns inverse matrix of eigen vectors(matrix T^-1)"""
        n_1, n_2 = n_vec[0], n_vec[1]
        press, rho, vel_x, vel_y = get_var_w(w_ij)
        if(a == None):
            a = air_vel(press, rho)
        div_const = 1/(2*(a**2))
        vel_n = vel_x * n_1 + vel_y * n_2
        vel_mag_2 = vel_x**2 + vel_y**2
        return div_const * np.array([[0.5*(gamma -1)*vel_mag_2 + a*vel_n, -a*n_1 - (gamma-1)*vel_x, -a*n_2 - (gamma-1)*vel_y, gamma -1],
                                    [2*(a**2) - (gamma - 1)*vel_mag_2, 2*(gamma - 1)*vel_x, 2*(gamma - 1)*vel_y, -2*(gamma - 1)],
                                    [2*(a**2)*(vel_y*n_1 - vel_x*n_2), 2*(a**2)*n_2, -2*(a**2)*n_1, 0],
                                    [0.5*(gamma - 1)*vel_mag_2 - a*vel_n, a*n_1 - (gamma -1)*vel_x, a*n_2 - (gamma - 1)*vel_y, gamma - 1]], dtype=float)
    
    def calc_D_mat(w_ij, n_vec, a = None)->np.array:
        """function that returns diagonal matrix of eigen values"""
        n_1, n_2 = n_vec[0], n_vec[1]
        press, rho, vel_x, vel_y = get_var_w(w_ij)
        if(a == None):
            a = air_vel(press, rho)
        vel_n = vel_x * n_1 + vel_y * n_2
        return np.diag([vel_n - a, vel_n, vel_n, vel_n + a])
    
    def calc_eig_vals(w_ij, n_vec, a = None)->np.array:
        "function that returns eigen values"
        n_1, n_2 = n_vec[0], n_vec[1]
        press, rho, vel_x, vel_y = get_var_w(w_ij)
        if(a == None):
            a = air_vel(press, rho)
        vel_n = vel_x * n_1 + vel_y * n_2
        return np.array([vel_n - a, vel_n, vel_n, vel_n + a])
    
    
    def calc_Roe_HH_flow(w_l, w_r, n_vec)->np.array:
        """function that returns Roe's numeric flow with Harten-Hyman entropy fix and lambda_max"""
        n_1, n_2 = n_vec[0], n_vec[1]
        press_l, rho_l, vel_x_l, vel_y_l = get_var_w(w_l)
        press_r, rho_r, vel_x_r, vel_y_r = get_var_w(w_r)
        H_l, H_r = entalpy(press_l, rho_l, vel_x_l, vel_y_l), entalpy(press_r, rho_r, vel_x_r, vel_y_r)
        rho_sqr_sum = math.sqrt(rho_l) + math.sqrt(rho_r)
        rho_hat = (0.5*(rho_sqr_sum))**2
        vel_x_hat = (math.sqrt(rho_l)*vel_x_l + math.sqrt(rho_r)*vel_x_r)/(rho_sqr_sum)
        vel_y_hat = (math.sqrt(rho_l)*vel_y_l + math.sqrt(rho_r)*vel_y_r)/(rho_sqr_sum)
        H_hat = (math.sqrt(rho_l)*H_l + math.sqrt(rho_r)*H_r)/(rho_sqr_sum)
        E_hat = rho_hat*H_hat/gamma + (gamma - 1)*rho_hat*(vel_x_hat**2 + vel_y_hat**2)/(2*gamma)
        a_hat = math.sqrt((gamma - 1)*(H_hat - 0.5*(vel_x_hat**2 + vel_y_hat**2)))
        w_hat = np.array([rho_hat, rho_hat*vel_x_hat, rho_hat*vel_y_hat, E_hat])
        # gamma_hat = T^-1 * (w_r - w_l)
        gamma_hat = calc_T_inv_mat(w_hat, n_vec, a_hat).dot(w_r - w_l)
        lambdas_hat = calc_eig_vals(w_hat, n_vec, a_hat)
        T_w_hat = calc_T_mat(w_hat, n_vec, a_hat)

        #alternative calculations of flows
        f_1_l = np.array([rho_l*vel_x_l, rho_l*(vel_x_l**2) + press_l, rho_l * vel_x_l * vel_y_l, (w_l[3] + press_l)*vel_x_l])
        f_2_l = np.array([rho_l*vel_y_l, rho_l*(vel_x_l* vel_y_l), rho_l * (vel_y_l**2) + press_l, (w_l[3] + press_l)*vel_y_l])
        f_1_r = np.array([rho_r*vel_x_r, rho_r*(vel_x_r**2) + press_r, rho_r * vel_x_r * vel_y_r, (w_r[3] + press_r)*vel_x_r])
        f_2_r = np.array([rho_r*vel_y_r, rho_r*(vel_x_r* vel_y_r), rho_r * (vel_y_r**2) + press_r, (w_r[3] + press_r)*vel_y_r])
        P_l = f_1_l * n_1 + f_2_l * n_2
        P_r = f_1_r * n_1 + f_2_r * n_2
        if(lambdas_hat[1] > 0):
            w_l_star = w_l + gamma_hat[0] * T_w_hat[:,0]
            lam_1_l = calc_eig_vals(w_l, n_vec)[0]
            lam_1_star = calc_eig_vals(w_l_star, n_vec)[0]
            lam_1_tilda = lam_1_l * (lam_1_star - lambdas_hat[0]) / (lam_1_star - lam_1_l) if (lam_1_l < 0 and lam_1_star > 0) else lambdas_hat[0]
            # T * D * T^-1 (w_l,n_vec) + gamma_hat_1 * (lambda_tilde_1^-) * r_1(w_hat)
            #return (calc_T_mat(w_l, n_vec) @ calc_D_mat(w_l, n_vec) @ calc_T_inv_mat(w_l, n_vec)).dot(w_l) + gamma_hat[0]*min(lam_1_tilda,0)* T_w_hat[:,0], abs(vel_x_hat*n_1 + vel_y_hat*n_2) + a_hat
            return P_l + gamma_hat[0]*min(lam_1_tilda,0)* T_w_hat[:,0], abs(vel_x_hat*n_1 + vel_y_hat*n_2) + a_hat
        else:
            w_r_star = w_r - gamma_hat[3] * T_w_hat[:,3]
            lam_4_r = calc_eig_vals(w_r, n_vec)[3]
            lam_4_star = calc_eig_vals(w_r_star, n_vec)[3]
            lam_4_tilda = lam_4_r * (lambdas_hat[3]- lam_4_star) / (lam_4_r - lam_4_star) if (lam_4_star < 0 and lam_4_r > 0) else lambdas_hat[3]
            # T * D * T^-1 (w_r,n_vec) + gamma_hat_4 * (lambda_tilde_4^+) * r_4(w_hat), calculation of lambda_max
            #return (calc_T_mat(w_r, n_vec) @ calc_D_mat(w_r, n_vec) @ calc_T_inv_mat(w_r, n_vec)).dot(w_r) - gamma_hat[3]*max(lam_4_tilda,0)* T_w_hat[:,3], abs(vel_x_hat*n_1 + vel_y_hat*n_2) + a_hat
            return P_r - gamma_hat[3]*max(lam_4_tilda,0)* T_w_hat[:,3], abs(vel_x_hat*n_1 + vel_y_hat*n_2) + a_hat
        
    #loading initial condition
    w_curr = np.empty((4, Nx_div + 2, Ny_div + 2), dtype = float)
    #calculate w_0 for t = 0
    for i in range(1, Nx_div + 1):
        x = net_x_cent[i]
        for j in range(1, Ny_div + 1):
            y = net_y_cent[j]
            if ((x**2 + y**2) <= (radius**2)):
                w_curr[:,i,j] = [rho_ins, rho_ins * vel_x_ins, rho_ins * vel_y_ins, press_ins/(gamma - 1) + 0.5*rho_ins*(vel_x_ins**2 + vel_y_ins**2)]
            else:
                w_curr[:,i,j] = [rho_out, rho_out * vel_x_out, rho_out * vel_y_out, press_out/(gamma - 1) + 0.5*rho_out*(vel_x_out**2 + vel_y_out**2)]

    #loading border condition
    w_curr[:,0,1:(Ny_div+1)] = w_curr[:,1,1:(Ny_div+1)]
    w_curr[:,-1,1:(Ny_div+1)] = w_curr[:,-2,1:(Ny_div+1)]
    w_curr[:,0:(Nx_div+2),0] = w_curr[:,0:(Nx_div+2),1]
    w_curr[:,0:(Nx_div+2),-1] = w_curr[:,0:(Nx_div+2),-2]

    w_whole = [np.copy(w_curr)] #saving values of w_0

    w_next = np.empty((4, Nx_div + 2, Ny_div + 2), dtype = float)

    indexs = {"left":([-1,0], [0,0]), "right":([0,0], [1,0]), "top":([0,0],[0,1]), "bottom":([0,-1], [0,0])}
    dirs = {"left":[1,0], "right":[1,0], "top":[0,1], "bottom":[0,1]}
    calculate = True
    #calculation of w_j^(n+1) for n+1 time
    while(calculate):
        max_lam_list = []
        H_2d_arr = np.empty((Nx_div, Ny_div), dtype=dict)
        #calculating flows and lambda_ij
        for i in range(1, Nx_div + 1):
            for j in range(1,Ny_div + 1):
                loc_lam_list = []
                H_dic = dict()
                for key, indx in indexs.items():
                    indx_1, indx_2 = indx
                    H_dic[key], loc_lambda = calc_Roe_HH_flow(w_curr[:,i + indx_1[0], j + indx_1[1]], w_curr[:,i + indx_2[0], j + indx_2[1]],dirs[key]) 
                    loc_lam_list.append(loc_lambda)
                max_lam_list.append(max(loc_lam_list))
                H_2d_arr[i-1][j-1] = H_dic

        time_step = calc_time_step(max_lam_list)
        #checks if we would step over time interval
        if(time_len - curr_t < 10**(-6)):
            time_step = time_len - curr_t
            curr_t = time_len
            calculate = False
        time_list.append(curr_t)
        curr_t += time_step

        for i in range(1, Nx_div + 1):
            for j in range(1,Ny_div + 1):
                # w_ij^(n+1) = w_ij^n - (tau_n/|D_ij|)*[h_y * (H_r(w_ij^n, w_(i+1,j)^n) + H_l(w_(i-1,j)^n, w_ij^n)) + h_x*(H_t(w_ij^n, w_(i,j+1)^n) + H_d(w_(i,j-1)^n, w_ij^n))], H_r, H_l, H_t, H_d corresponds to directions right, left, top and bottom
                w_next[:,i,j] = w_curr[:,i,j] - (time_step/(step_x * step_y))*(step_y * (H_2d_arr[i-1][j-1]["right"] - H_2d_arr[i-1][j-1]["left"]) + step_x * (H_2d_arr[i-1][j-1]["top"] - H_2d_arr[i-1][j-1]["bottom"]))

        #loading border condition
        w_next[:,0,1:(Ny_div+1)] = w_next[:,1,1:(Ny_div+1)]
        w_next[:,-1,1:(Ny_div+1)] = w_next[:,-2,1:(Ny_div+1)]
        w_next[:,0:(Nx_div+2),0] = w_next[:,0:(Nx_div+2),1]
        w_next[:,0:(Nx_div+2),-1] = w_next[:,0:(Nx_div+2),-2]
        
        w_curr = np.copy(w_next)
        w_whole.append(np.copy(w_next))

        #plotting data for debugging
        # rho_2D = np.empty((62,62))
        # vel_x_2D = np.empty((62,62))
        # vel_y_2D = np.empty((62,62))
        # press_2D = np.empty((62,62))
        # #calculating density, velocity and press from values of w
        # for i in range(62):
        #     for j in range(62):
        #         rho, vel_x, vel_y, press = get_var_w(w_next[:,i,j])
        #         rho_2D[i,j] = rho
        #         vel_x_2D[i,j] = vel_x
        #         vel_y_2D[i,j] = vel_y
        #         press_2D[i,j] = press
        # #data_pyth = [rho_2D, vel_x_2D, vel_y_2D, press_2D]
        # x_grid, y_grid = np.meshgrid(net_x_cent, net_y_cent)
        # #plotting python data
        # data_pyth = [rho_2D, vel_x_2D, vel_y_2D, press_2D]
        # #plotting python data
        # titles = ["Python density", "Python velocity in x direction", "Python velocity in y direction", "Python pressure"]
        # for i in range(4):
        #     fig = plt.figure()
        #     plt.title(titles[i])
        #     ax = fig.add_subplot(projection="3d")
        #     ax.plot_surface(x_grid, y_grid, data_pyth[i])
        #     plt.show() 

    return net_x_cent, net_y_cent, time_list, w_whole

def plot_Roe_HH(x, y, t, w_whole):
    """Function that plots sollution of Euler's equations in 2D and compares it with function from matlab for prescribed conditions."""
    #x, y, t, w_whole = Roe_HH_2D(x_inter=(-1,1), y_inter=(-1,1), rho_ins=1, vel_x_ins=0, vel_y_ins=0, press_ins=1, rho_out=0.125,
    #                       vel_x_out=0, vel_y_out=0, press_out=0.1, radius=0.5, time_len=0.2, cons_C=0.85, gamma=1.4, Nx_div=60, Ny_div=60)
    def get_var_w(w_ij)->tuple:
        """funtion that returns density, velocity and pressure from vector of w"""
        rho = w_ij[0]
        vel_x = w_ij[1]/rho
        vel_y = w_ij[2]/rho
        press = (1.4 - 1)*(w_ij[3] - 0.5*rho*(vel_x**2 + vel_y**2))
        return press, rho, vel_x, vel_y
    
    rho_2D = np.empty((62,62))
    vel_x_2D = np.empty((62,62))
    vel_y_2D = np.empty((62,62))
    press_2D = np.empty((62,62))
    #calculating density, velocity and press from values of w at time 0.2
    w_last = w_whole[-1]
    for i in range(62):
        for j in range(62):
            press, rho, vel_x, vel_y = get_var_w(w_last[:,i,j])
            rho_2D[i,j] = rho
            vel_x_2D[i,j] = vel_x
            vel_y_2D[i,j] = vel_y
            press_2D[i,j] = press
        
    #loading matlab data
    rho_data = pd.read_excel("mat_saves/2D_rho.xlsx", header = None, dtype=float).to_numpy()
    vel_x_data = pd.read_excel("mat_saves/2D_vel_x.xlsx", header = None, dtype=float).to_numpy()
    vel_y_data = pd.read_excel("mat_saves/2D_vel_y.xlsx", header = None, dtype=float).to_numpy()
    press_data = pd.read_excel("mat_saves/2D_press.xlsx", header = None, dtype=float).to_numpy()
    x_grid, y_grid = np.meshgrid(x, y)
    data_mat = [rho_data, vel_x_data, vel_y_data, press_data] 

    #plotting matlab data
    titles = ["Matlab density", "Matlab velocity in x direction", "Matlab velocity in y direction", "Matlab press"]
    for i in range(4):
        fig = plt.figure()
        plt.title(titles[i])
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(x_grid, y_grid, data_mat[i])
        plt.show()

    data_pyth = [rho_2D, vel_x_2D, vel_y_2D, press_2D]
    #plotting python data
    titles = ["Python density", "Python velocity in x direction", "Python velocity in y direction", "Python pressure"]
    for i in range(4):
        fig = plt.figure()
        plt.title(titles[i])
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(x_grid, y_grid, data_pyth[i])
        plt.show()     
    
