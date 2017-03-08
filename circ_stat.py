# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 00:09:07 2016

@author: Vladislav
"""

from __future__ import division, print_function, absolute_import
import cmath
import numpy as np
from scipy import integrate, special
from math import pi, sin, cos, exp
#from sympy import summation, factorial, oo, symbols
from scipy.optimize import brentq
from scipy.misc import derivative

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(1, 1)

class circ_stat():
    def __init__ (self, a = None, b = None):
#        self.a = a
#        self.b = b
        if a is None:
            self.a = 0
        else:
            self.a = a
        if b is None:
            self.b = 2*pi
        else:
            self.b = b
#        self.a = a % (2*pi) 
#        self.b = b % (2*pi)
            
#    def c_mom_func(self, x, m):
#        return cos(m*x)*self.pdf(x)
        
#    def s_mom_func(self, x, m):
#        return sin(m*x)*self.pdf(x)
            
    def mom_func(self, x, m):
        return complex(cos(m*x)*self.pdf(x), sin(m*x)*self.pdf(x))
            
    def mth_moment(self, m):
        """
            Mth angular moment.
            
            m_{n}=\int_{\Gamma }pdf(x)e^{ixn}dx
            
            e^{ixn} = cos(xn) + isin(nx)
            
            These two integrands counted in c_mom_func
            and s_mom_func respectively.
            
            Direction is arctan(s_mom/c_mom)
        """
        def real(x, m):
            return self.mom_func(x, m).real
            
        def img(x, m):
            return self.mom_func(x, m).imag
            
        c_mom = integrate.quad(real, self.a, self.b, m)[0]
        s_mom = integrate.quad(img, self.a, self.b, m)[0]
        
        return complex(c_mom, s_mom)
    
    def mean(self):
        return cmath.phase(self.mth_moment(1))
        
    def dispersion(self):
        return abs(self.mth_moment(1))
        
        
    def pdf(self, x, loc = 0, scale = 1, *args):
        """ Probability density function.
		Defided as function that satisfies for 3 purposes:
		1) f(x) > 0
		2) \int_{0}^{2\pi }f(x)dx = 1
		3) f(x+2\pi k) = f(x)
		
		"Topics in Circular Statistics"
		S. R. Jammalamadaka and A. SenGupta  """ 
        loc = loc%(self.b - self.a)
        x = np.asarray(x)
        cond1 = (scale > 0) & ((x-loc)/scale >= self.a) & ((x-loc)/scale <= self.b)
#        np.putmask(output, type(x) == type(np.nan), np.nan)
        output = np.zeros(np.shape(cond1), 'd')
        np.putmask(output, np.isnan(x), np.nan)
        if np.any(cond1):
            np.place(output, cond1, self._pdf(x, *args)/scale)
        output /= integrate.quad(self._pdf, self.a, self.b)[0]
        return output 
            
    def cdf(self, x, loc = 0, scale = 1, *args):
        """
		Cumulative distribution function.
		
		1) F(x) = Pr(0 < \theta \leq x), 0 \leq x\leq 2\pi 
		2) F(x+2\pi) - F(x) = 1, -\infty <x< \infty
		
		"Directional Statistics"
		Kanti V. Mardia, Peter E. Jupp
            """
        loc = loc%(self.b - self.a)
        x = np.asarray(x)
        cond1 = (scale > 0) & ((x-loc)/scale >= self.a) & ((x-loc)/scale <= self.b)
        output = np.zeros(np.shape(cond1), 'd')
        np.putmask(output, np.isnan(x), np.nan)
        y = x//(2*pi)
        x = x%(2*pi)
        if np.any(cond1):
            np.place(output, cond1, self._cdf(x, *args)/scale)
        output /= integrate.quad(self._pdf, self.a, self.b)[0]
        return output + y 
        
    def _cdf(self, x):
        x = np.asarray(x)
        for k in np.nditer(x, op_flags=['readwrite']):
            k[...] = integrate.quad(self._pdf, self.a, k)[0]        
        return x
        
    def _pdf(self, x):
        x = np.asarray(x)
        return derivative(self._cdf, x, dx=1e-6)
            
    def ppf(self, q, *args, loc = 0, scale = 1):
        """
            Percentage point function.
        """
        loc = loc%(self.b - self.a)
        
        return self._ppf(q, *args, loc, scale)
        
    def _ppf(self, q, *args, loc=0, scale=1):
        def Q(y, z):
            return self.cdf((y-loc)/scale) - z
            
        xi = brentq(Q, self.a, self.b, args=(q,))
        return xi
        
    def rvs(self, *args, size=1, random_state=None):
        """
            Generates random numbers of a given distribution
        """
        s = size
        return self._rvs(*args, size=s, random_state=None)        
        
    def _rvs(self, *args, size=1, random_state=None):
        q = np.random.random_sample(size)
        def Q(y, z):
            return self.cdf(y) - z
        
        for x in np.nditer(q, op_flags=['readwrite']):
            x[...] = self.ppf(x)
            
        return q
        
    def median(self, *args, loc = 0, scale = 1):
        loc = loc%(2*pi)
        l = loc
        return self.ppf(0.5, loc = l)
    
    def entropy(self):
        """
            Mardia, p.68
        """
        def integrand(x):
            return -self.pdf(x)*np.log(self.pdf(x))
        return integrate.quad(integrand, self.a, self.b)[0]            
        
            
class circ_uniform(circ_stat):
    """
		Uniform distribution.
  
           pdf(x) = const = 1/(b-a)
    """
    def _pdf(self, x):
        return 1/(2*pi)
        
    def _cdf(self, x):
        """Интеграл по одному обороту даёт 1, соответственно
        _cdf(x) = x//(2*pi)+x%(2*pi) = x/(2*pi)"""
        return x/(2*pi)
        
unif = circ_uniform()

class circ_cardioid(circ_stat):
    """
        Cardioid distribution.
  
        pdf(x) was defined at p. 34 of
        "Topics in Circular Statistics"
	   S. R. Jammalamadaka and A. SenGupta.
    
        cdf(x) was calculated manually.
        
        Note that:
        \mu \in \left [ 0, 2\pi  \right ),
        \rho \in \left ( -0.5, 0.5 \right )
    """
#    def _pdf(self, x, rho = 0.33, mu = 0):
#        return (1 + 2*rho*np.cos(x-mu))/(2*pi)

    def _pdf(self, x, rho = 0.33):
#        print(x)
        return (1 + 2*rho*np.cos(x))/(2*pi)   
        
    def _cdf(self, x, rho = 1, mu = 0):
        return (x%(2*pi))/(2*pi) + rho*(np.sin(mu)-np.sin(mu-(x%(2*pi))))/(2*pi)
        
card = circ_cardioid()
card1 = circ_cardioid(1, 4)

class circ_von_mises(circ_stat):
    """
		Von Mises distribution (Circular Normal distribution).
          
           defined at p. 35 of 
           "Topics in Circular Statistics"
	      S. R. Jammalamadaka and A. SenGupta.
    """
#    def I_0(self, kappa, mem = 50):
#        """Modified Bessel Function of the first kind and order zero"""
#        r = symbols('r')
#        return summation((kappa/2)**(2*r)*(1/factorial(r))**2, (r, 0, oo))
        
        
    def I_v(self, v, x):
        return special.iv(v, x)
		
    def _pdf(self, x, kappa = 1):
        return np.exp(kappa*np.cos(x))/(2*pi*self.I_v(0, kappa))
	
#    def _cdf(self, x, mu = 0, kappa = 1):
#        p = symbols('p')
#        return (x*self.I_0(kappa)+2*summation(self.I_v(p, kappa)*np.sin(p*(x-mu))/p, (p, 1, oo)))/(2*pi*self.I_0(kappa))
        
mises = circ_von_mises()

class circ_normal(circ_stat):
    """
        Wrapped normal distribution.
        
        Mardia, p.58
    """
    def _pdf(self, x, wraps = 50):
#        fg = x
#        print(fg)
        dens = 0
        for k in range(-wraps, wraps):
            dens += np.exp(-(x+2*pi*k)**2/2)
        dens /= (2*pi)
        return dens

class circ_cauchy(circ_stat):
    """
        Wrapped Cauchy distribution
    """
    def _pdf(self, x, a = 1, wraps = 50):
        dens = 0
        for k in range(-wraps, wraps):
            dens += a/(pi*(a**2+x**2))
        return dens

        

norm = circ_normal()
cauchy = circ_cauchy()

def pearson_mises(angles, d, mu, kappa):
    """
        
        Pearson's chi-square test of Mises distribution.
        angles - list of angles
        d - degrees of freedom
        mu, kappa - parameters of Mises distribution        
        
    """
    n = len(angles)
    borders = []
    for o in range(d):
        borders.append(o*2*pi/d)
    borders.append(2*pi)
    estimated = []
    observed = []
    for o in range(d):
        estimated.append(integrate.quad(mises.pdf, borders[o], borders[o+1], args = (mu, kappa))[0])
        observed.append(0)
#        estimated.append(mises.cdf)
    for i in angles:
        for j in range(d):
            if i > borders[j] and i < borders[j+1]:
                observed[j] += 1
    for i in range(d):
        observed[i] /= n
    chi2 = 0
    for i in range(d):
        chi2 += ((observed[i]-estimated[i])**2/estimated[i])
    chi2 *= n
    return chi2
    

def krmtest(observed):
    """
        Kramers-Rao point estimation of Mises distribution.
        Returns Pearson's chi-square statistics.
        
        Mardia, p. 126-136
    """
    observed = observed % (2*pi)
    n = len(observed)
    c_avg = 0
    s_avg = 0
    r_avg = 0
    
    for i in observed:
        c_avg += cos(i)
        s_avg += sin(i)
        
    c_avg /= n
    s_avg /= n
    r_avg = (c_avg**2+s_avg**2)**(1/2)
    mu = np.arctan(s_avg/c_avg)

    if r_avg < 0.45:
        kappa = r_avg*(12+6*r_avg**2+5*r_avg**4)
    elif r_avg > 0.8:
        kappa = 1/(2*(1-r_avg)-(1-r_avg)**2-(1-r_avg)**3)
    elif r_avg >= 0.45 and r_avg < 0.6:
        kappa = 1.0122 + (r_avg-0.45)*(1.51574-1.01022)/0.15
    elif r_avg >= 0.6 and r_avg < 0.7:
        kappa = 1.5174 + (r_avg-0.6)*(2.01363-1.51574)/0.1
    else:
        kappa = 2.01363 + (r_avg-0.7)*(2.87129-2.01363)/0.1
    
    return pearson_mises(observed, 9, mu, kappa)
    




#krmtest() 
#a=pearson_mises(np.array([8, 3, 3.1]), 7, 5, 1)
#print(a)
#a = np.array([1, 2, 3])
#b = mises.pdf(a)
#print(b)
#a6 = norm.cdf(2*pi)
#print(a6)

#a7 = cauchy._median()
#a7 = card.entropy()
#a7 = mises.dispersion()
#print(a7)

#arr = np.array([1, 10000])
#arr = [1, 3.0]
#print(arr)
#a1 = unif.cdf(arr, 0., 1.)
#a1 = unif.mth_moment(2)
#a2 = card.cdf(arr)
#a3 = card.mth_moment(1)a
#a4 = unif._median()
#print(a4)
#print(a3)
#a3 = card._median()
#print(a3)
    
#a3 = mises.pdf(6.27)
#a3 = mises._median()
#print(a3)   
    
x = np.linspace(0.01, 16.28, 628)
#x = [1, 2]
#www = norm.pdf(x)
#print(type(www))
#y = mises.pdf(x)
    
#ax.plot(x, norm.pdf(x, 1, 1), 'r-', lw=5, alpha=0.6, label='norm pdf')
#print(norm.pdf([0.5, 7], 1, 1))
#summ = 0
#for k in x:
#    summ += norm.cdf(k, 2, 2)*0.01
#print(summ)
#print(norm.cdf(6.28))
    
#print(card.pdf([1, 2, np.nan]))
#print([card.pdf(_) for _ in [1, 2, 3, 4]])
#print(card.rvs(size=(10, 2)))
#print(card.median(loc = 20))
#print(card.rvs(size=(2, 10)))    
#print(card.ppf(0.5))
#a = card.pdf([1, 2, 3 ], loc = 0.5)
#print(a)
#print(norm.pdf([1, 2, 2, 1], loc=1))