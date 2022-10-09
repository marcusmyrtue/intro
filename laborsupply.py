import numpy as np
from scipy import optimize 

def implied_tax(l,w,tau0,tau1,kappa):
    """ Calculates the implied tax rate of labor supply choice

    Args: 

        l (float): labor supply
        w (float): wage
        tau0 (float): standard labor tax
        tau1 (float): top bracket tax
        kappa (float): top bracket tax income threshold

    Returns:

        (float): Total tax bill

    """
    return tau0*l*w + tau1*np.fmax(w*l - kappa, 0) 

def implied_c(l,m,w,tau0,tau1,kappa):
    """ Calculates the implied consumption of labor supply choice

    Args:

        l (float): labor supply
        w (float): wage
        tau0 (float): standard labor tax
        tau1 (float): top bracket tax
        kappa (float): top bracket tax income threshold

    Returns:

        (float): consumption

    """

    return m + w*l - implied_tax(l,w,tau0,tau1,kappa)

def utility(c,l,nu,frisch):
    """ Calculates the utility from consumption and labor supply choice

    Args: 

        c (float): consumption
        l (float): labor supply
        nu (float): disutility of labor supply
        frisch (frisch): frisch elasticity of labor supply

    Returns: 

        (float): utility 
    """
    return np.log(c) - nu*l**(1 + 1/frisch) / (1 + 1/frisch) 

def value_of_choice(l,nu,frisch,m,w,tau0,tau1,kappa):
    """ Calculates implied utility by of consumption and labor supply choice

    Args: 

        l (float): labor supply
        nu (float): disutility of labor supply
        frisch (frisch): frisch elasticity of labor supply
        m (float): cash on hand
        w (float): wage
        tau0 (float): standard labor tax
        tau1 (float): top bracket tax
        kappa (float): top bracket tax income threshold

    Returns:

        (float): utility

    """
    
    c = implied_c(l,m,w,tau0,tau1,kappa)
    return utility(c,l,nu,frisch) 

def find_optimal_labor_supply(nu,frisch,m,w,tau0,tau1,kappa):
    """ Finds optimal labor supply from supplied inputs

    Args:

        nu (float): disutility of labor supply
        frisch (frisch): frisch elasticity of labor supply
        m (float): cash on hand
        w (float): wage
        tau0 (float): standard labor tax
        tau1 (float): top bracket tax
        kappa (float): top bracket tax income threshold

    Returns:

        (float): labor supply
    """

    obj = lambda l: -value_of_choice(l,nu,frisch,m,w,tau0,tau1,kappa)
    res = optimize.minimize_scalar(obj,bounds=(1e-8,1),method='bounded')
    return res.x


def tax_revenue(nu,frisch,m,w,tau0,tau1,kappa):
    """ Calculates total tax revenue from 
    
    Args:
    
        nu (float): disutility of labor supply
        frisch (frisch): frisch elasticity of labor supply
        m (float): cash on hand
        w (float): wage
        tau0 (float): standard labor tax
        tau1 (float): top bracket tax
        kappa (float): top bracket tax income threshold
        
    Returns:
    
        (float): total tax revenue
    
    """

    # a. optimal labor supply
    N = w.size 
    l_vec = np.zeros(N) 

    for i in range(N):
        l_vec[i] = find_optimal_labor_supply(nu,frisch,m,w[i],tau0,tau1,kappa)


    # b. taxes
    T = np.sum(implied_tax(l_vec,w,tau0,tau1,kappa))
    
    return T 
    

def obj(x,nu,frisch,m,w):
    """ Calculates the objective function to be minimized 

    Args:

        x (np.array): tax parameters
        nu (float): disutility of labor supply
        frisch (frisch): frisch elasticity of labor supply
        m (float): cash on hand
        w (float): wage

    Return:

        (float): negative tax revenue

    """

    tau0 = x[0]
    if x.size > 1:
        tau1 = x[1]
        kappa = x[2]
    else:
        tau1 = 0.0
        kappa = 0.0

    T = tax_revenue(nu,frisch,m,w,tau0,tau1,kappa)  

    print(f'tau0 = {tau0:10.8f}, tau1 = {tau1:10.8f}, kappa = {kappa:10.8f} -> T = {T:12.8f},') 

    return -T 



