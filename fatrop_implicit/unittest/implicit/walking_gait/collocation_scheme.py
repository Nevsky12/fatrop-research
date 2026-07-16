"""
CollocationScheme

This script returns matrices needed for setting the collocation problem.
This function is adapted from an example (direct_collocation.m) from the
CasADi example pack.

INPUT:
    d : int
        degree of the interpolating polynomial
    method : str
        'radau' or 'legendre'

OUTPUT:
    tau_root : array
        Collocation points
    C : array
        Coefficients of the collocation equation
    D : array
        Coefficients of the continuity equation
    B : array
        Coefficients of the quadrature function

Original author: Antoine Falisse
Original date: 12/19/2018

Converted to Python: 2026
"""

import numpy as np
import casadi


def collocation_scheme(d, method):
    """
    Return collocation matrices for direct collocation methods.
    
    Parameters
    ----------
    d : int
        Degree of the interpolating polynomial
    method : str
        Collocation method: 'radau' or 'legendre'
    
    Returns
    -------
    tau_root : np.ndarray
        Collocation points (shape: (d+1,))
    C : np.ndarray
        Coefficients of the collocation equation (shape: (d+1, d+1))
    D : np.ndarray
        Coefficients of the continuity equation (shape: (d+1, 1))
    B : np.ndarray
        Coefficients of the quadrature function (shape: (d+1, 1))
    """
    
    # Get collocation points
    tau_root_base = casadi.collocation_points(d, method)
    tau_root = np.concatenate([[0], tau_root_base])
    
    # Coefficients of the collocation equation
    C = np.zeros((d + 1, d + 1))
    
    # Coefficients of the continuity equation
    D = np.zeros((d + 1, 1))
    
    # Coefficients of the quadrature function
    B = np.zeros((d + 1, 1))
    
    # Construct polynomial basis
    for j in range(d + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the
        # collocation point
        coeff = np.array([1.0])
        for r in range(d + 1):
            if r != j:
                coeff = np.polymul(coeff, np.array([1, -tau_root[r]]))
                coeff = coeff / (tau_root[j] - tau_root[r])
        
        # Evaluate the polynomial at the final time to get the coefficients of
        # the continuity equation
        D[j, 0] = np.polyval(coeff, 1.0)
        
        # Evaluate the time derivative of the polynomial at all collocation
        # points to get the coefficients of the collocation equation
        pder = np.polyder(coeff)
        for r in range(d + 1):
            C[j, r] = np.polyval(pder, tau_root[r])
        
        # Evaluate the integral of the polynomial to get the coefficients of
        # the quadrature function
        pint = np.polyint(coeff)
        B[j, 0] = np.polyval(pint, 1.0)
    
    return tau_root, C, D, B
