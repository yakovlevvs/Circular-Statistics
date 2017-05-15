"""
Created on Fri Apr  7 15:35:19 2017

@author: Vladislav
"""

import sys
import keyword
import re
import types

from scipy._lib.six import string_types, exec_

from scipy._lib._util import getargspec_no_self as _getargspec

from scipy import integrate, special

from scipy._lib.six import string_types

from numpy import (arange, putmask, ravel, take, ones, sum, shape,
                   product, reshape, zeros, floor, logical_and, log, sqrt, exp,
                   ndarray)
                   
from scipy import optimize
                   
from math import sin, cos

from numpy import (place, any, argsort, argmax, vectorize,
                   asarray, nan, inf, isinf, NINF, empty)

import numpy as np

from scipy.misc import derivative

from scipy.stats import _distn_infrastructure

from scipy.stats import norm, vonmises, dweibull

from scipy.optimize import brentq

from cmath import phase

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(1, 1)

try:
    from new import instancemethod
except ImportError:
    # Python 3
    def instancemethod(func, obj, cls):
        return types.MethodType(func, obj)

docdict = {
    
}

parse_arg_template = """
def _parse_args(self, %(shape_arg_str)s %(locscale_in)s):
    return (%(shape_arg_str)s), %(locscale_out)s
def _parse_args_rvs(self, %(shape_arg_str)s %(locscale_in)s, size=None):
    return (%(shape_arg_str)s), %(locscale_out)s, size
def _parse_args_stats(self, %(shape_arg_str)s %(locscale_in)s, moments='mv'):
    return (%(shape_arg_str)s), %(locscale_out)s, moments
"""


def argsreduce(cond, *args):
    """Return the sequence of ravel(args[i]) where ravel(condition) is
    True in 1D.
    Examples
    --------
    >>> import numpy as np
    >>> rand = np.random.random_sample
    >>> A = rand((4, 5))
    >>> B = 2
    >>> C = rand((1, 5))
    >>> cond = np.ones(A.shape)
    >>> [A1, B1, C1] = argsreduce(cond, A, B, C)
    >>> B1.shape
    (20,)
    >>> cond[2,:] = 0
    >>> [A2, B2, C2] = argsreduce(cond, A, B, C)
    >>> B2.shape
    (15,)
    """
    newargs = np.atleast_1d(*args)
    if not isinstance(newargs, list):
        newargs = [newargs, ]
    expand_arr = (cond == cond)
    return [np.extract(cond, arr1 * expand_arr) for arr1 in newargs]
    
def valarray(shape, value=nan, typecode=None):
    """Return an array of all value.
    """

    out = ones(shape, dtype=bool) * value
    if typecode is not None:
        out = out.astype(typecode)
    if not isinstance(out, ndarray):
        out = asarray(out)
    return out

class rv_circular():
    def __init__(self, a=None, b=None, xtol=1e-14,
                 badvalue=None, name=None, longname=None,
                 shapes=None, extradoc=None, seed=None):

#        super(rv_continuous, self).__init__(seed)

        # save the ctor parameters, cf generic freeze
        self._ctor_param = dict(
            a=a, b=b, xtol=xtol,
            badvalue=badvalue, name=name, longname=longname,
            shapes=shapes, extradoc=extradoc, seed=seed)

        if badvalue is None:
            badvalue = nan
        if name is None:
            name = 'Distribution'
        self.badvalue = badvalue
        self.name = name
        self.a = a
        self.b = b
        if a is None:
            self.a = 0
        if b is None:
            self.b = 2*np.pi
        self.xtol = xtol
        self._size = 1
        self.shapes = shapes
        self._construct_argparser(meths_to_inspect=[self._pdf, self._cdf],
                                  locscale_in='loc=0, scale=1',
                                  locscale_out='loc, scale')

        # nin correction
        self._ppfvec = vectorize(self._ppf_single, otypes='d')
        self._ppfvec.nin = self.numargs + 1
#        self.vecentropy = vectorize(self._entropy, otypes='d')
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self._cdfvec.nin = self.numargs + 1

        # backwards compat.  these were removed in 0.14.0, put back but
        # deprecated in 0.14.1:
#        self.vecfunc = np.deprecate(self._ppfvec, "vecfunc")
#        self.veccdf = np.deprecate(self._cdfvec, "veccdf")

        # Because of the *args argument of _mom0_sc, vectorize cannot count the
        # number of arguments correctly.
#        self.generic_moment.nin = self.numargs + 1

        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = "An "
            else:
                hstr = "A "
            longname = hstr + name

        if sys.flags.optimize < 2:
            # Skip adding docstrings if interpreter is run with -OO
            if self.__doc__ is None:
                self._construct_default_doc(longname=longname,
                                            extradoc=extradoc,
                                            docdict=docdict,
                                            discrete='continuous')
#            else:
#                self._construct_doc(docdict, dct.get(self.name))
                                            
    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.
        Returns condition array of 1's where arguments are correct and
         0's where they are not.
        """
        cond = 1
        for arg in args:
            cond = logical_and(cond, (asarray(arg) > 0))
        return cond
                                            
    def _construct_argparser(
            self, meths_to_inspect, locscale_in, locscale_out):
        """Construct the parser for the shape arguments.
        Generates the argument-parsing functions dynamically and attaches
        them to the instance.
        Is supposed to be called in __init__ of a class for each distribution.
        If self.shapes is a non-empty string, interprets it as a
        comma-separated list of shape parameters.
        Otherwise inspects the call signatures of `meths_to_inspect`
        and constructs the argument-parsing functions from these.
        In this case also sets `shapes` and `numargs`.
        """

        if self.shapes:
            # sanitize the user-supplied shapes
            if not isinstance(self.shapes, string_types):
                raise TypeError('shapes must be a string.')

            shapes = self.shapes.replace(',', ' ').split()

            for field in shapes:
                if keyword.iskeyword(field):
                    raise SyntaxError('keywords cannot be used as shapes.')
                if not re.match('^[_a-zA-Z][_a-zA-Z0-9]*$', field):
                    raise SyntaxError(
                        'shapes must be valid python identifiers')
        else:
            # find out the call signatures (_pdf, _cdf etc), deduce shape
            # arguments. Generic methods only have 'self, x', any further args
            # are shapes.
            shapes_list = []
            for meth in meths_to_inspect:
                shapes_args = _getargspec(meth)   # NB: does not contain self
                args = shapes_args.args[1:]       # peel off 'x', too

                if args:
                    shapes_list.append(args)

                    # *args or **kwargs are not allowed w/automatic shapes
                    if shapes_args.varargs is not None:
                        raise TypeError(
                            '*args are not allowed w/out explicit shapes')
                    if shapes_args.keywords is not None:
                        raise TypeError(
                            '**kwds are not allowed w/out explicit shapes')
                    if shapes_args.defaults is not None:
                        raise TypeError('defaults are not allowed for shapes')

            if shapes_list:
                shapes = shapes_list[0]

                # make sure the signatures are consistent
                for item in shapes_list:
                    if item != shapes:
                        raise TypeError('Shape arguments are inconsistent.')
            else:
                shapes = []

        # have the arguments, construct the method from template
        shapes_str = ', '.join(shapes) + ', ' if shapes else ''  # NB: not None
        dct = dict(shape_arg_str=shapes_str,
                   locscale_in=locscale_in,
                   locscale_out=locscale_out,
                   )
        ns = {}
        exec_(parse_arg_template % dct, ns)
        # NB: attach to the instance, not class
        for name in ['_parse_args', '_parse_args_stats', '_parse_args_rvs']:
            setattr(self, name,
                    instancemethod(ns[name], self, self.__class__)
                    )

        self.shapes = ', '.join(shapes) if shapes else None
        if not hasattr(self, 'numargs'):
            # allows more general subclassing with *args
            self.numargs = len(shapes)
                                            
    def mom_func(self, x, m, *args):
        return complex(cos(m*x)*self.pdf(x, args), sin(m*x)*self.pdf(x, args))
            
    def mth_moment(self, m, *args):
        """
            Mth angular moment.
            
            m_{n}=\int_{\Gamma }pdf(x)e^{ixn}dx
            
            e^{ixn} = cos(xn) + isin(nx)
            
            These two integrands counted in c_mom_func
            and s_mom_func respectively.
            
            Direction is arctan(s_mom/c_mom)
        """
        def real(x, m, *args):
            return self.mom_func(x, m, *args).real
            
        def img(x, m, *args):
            return self.mom_func(x, m, *args).imag
            
        args, loc, scale = self._parse_args(*args)
        a = args
        c_mom = integrate.quad(real, self.a, self.b, args = (m, a,) )[0]
        s_mom = integrate.quad(img, self.a, self.b, args = (m, a,))[0]
        
        return complex(c_mom, s_mom)
                
    def pdf(self, x, *args, **kwds):
        """
        Probability density function at x of the given RV.
        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)
        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x
        """
#        loc = asarray(loc)
#        scale = asarray(scale)
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        x = asarray((x-loc)*1.0/scale)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = (scale > 0) & (x >= self.a) & (x <= self.b)
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        putmask(output, (1-cond0)+np.isnan(x), self.badvalue)
        if any(cond):
            goodargs = argsreduce(cond, *((x,)+args+(scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._pdf(*goodargs) / scale)
        if output.ndim == 0:
            return output[()]
        return output
        
    def cdf(self, x, *args, **kwds):
        """
        Cumulative distribution function of the given RV.
        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)
        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `x`
        """
#        loc = asarray(loc)
#        scale = asarray(scale)
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        x = (x-loc)*1.0/scale
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = (scale > 0) & (x > self.a) & (x < self.b)
        cond2 = (x >= self.b) & cond0
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._cdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output        

#    def _cdf(self, x, *args):
#        x = np.asarray(x)
#        if args == ():
#            for k in np.nditer(x, op_flags=['readwrite']):
#                k[...] = integrate.quad(self._pdf, self.a, k, args = args)[0]        
#                return x 
#        else:
#            ar = np.asarray(args)
#            for k in np.nditer([x, ar], op_flags=['readwrite', 'readwrite']):
#                k[...] = integrate.quad(self._pdf, self.a, k[0], args = k[1])[0]        
#        return x 
    def _cdf_single(self, x, *args):
        return integrate.quad(self._pdf, self.a, x, args=args)[0]

    def _cdf(self, x, *args):
        return self._cdfvec(x, *args)
        
            
        
    def _pdf(self, x, *args):
        x = np.asarray(x)
        return derivative(self._cdf, x, dx=1e-6)
        
        
    def ppf(self, q, *args, **kwds):
        """
        Percent point function (inverse of `cdf`) at q of the given RV.
        Parameters
        ----------
        q : array_like
            lower tail probability
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)
        Returns
        -------
        x : array_like
            quantile corresponding to the lower tail probability q.
        """
        args, loc, scale = self._parse_args(*args, **kwds)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
#        q = asarray(q)
#        loc = asarray(loc)
        scale = asarray(scale)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 0)
        cond3 = cond0 & (q == 1)
        cond = cond0 & cond1
        output = valarray(shape(cond), value=self.badvalue)

        lower_bound = self.a * scale + loc
        upper_bound = self.b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        if any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((q,)+args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            place(output, cond, self._ppf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output
        
#    def _ppf(self, q, *args):
#        def Q(y, z):
#            return self.cdf((y-loc)/scale) - z
#            return self.cdf(*(y, )+args)-z
            
#        for x in np.nditer(q, op_flags=['readwrite']):
#            x[...] = brentq(Q, self.a, self.b, args=(x,))
            
#        xi = brentq(Q, self.a, self.b, args=(q,))
#        return q

    def _ppf(self, q, *args):
        return self._ppfvec(q, *args)

    
    def _ppf_to_solve(self, x, q, *args):
        return self.cdf(*(x, )+args)-q

    def _ppf_single(self, q, *args):
        left = right = None
        if self.a > -np.inf:
            left = self.a
        if self.b < np.inf:
            right = self.b

        factor = 10.
        if not left:  # i.e. self.a = -inf
            left = -1.*factor
            while self._ppf_to_solve(left, q, *args) > 0.:
                right = left
                left *= factor
            # left is now such that cdf(left) < q
        if not right:  # i.e. self.b = inf
            right = factor
            while self._ppf_to_solve(right, q, *args) < 0.:
                left = right
                right *= factor
            # right is now such that cdf(right) > q

        return optimize.brentq(self._ppf_to_solve,
                               left, right, args=(q,)+args, xtol=self.xtol)
        
    def rvs(self, size=1, random_state=None,  *args, **kwds):
        """
            Generates random numbers of a given distribution
        """
        s = size
        return self._rvs(size=s, random_state=None, *args, **kwds)        
        
    def _rvs(self, size=1, random_state=None, *args, **kwargs):
        q = np.random.random_sample(size)
#        def Q(y, z, *args, **kwargs):
#            return self.cdf(y, *args, **kwargs) - z
        
        for x in np.nditer(q, op_flags=['readwrite']):
            x[...] = self.ppf(x, *args, **kwargs)
            
        return q
        
    def median(self, *args):
        return self.ppf(0.5)
    
    def entropy(self, *args):
        """
            Mardia, p.68
        """
        def integrand(x):
            return -self.pdf(x, args)*np.log(self.pdf(x, args))
        return integrate.quad(integrand, self.a, self.b)[0] 
        
    def fit(self, data, *args, **kwds):
        Narg = len(args)
        if Narg > self.numargs:
            raise TypeError("Too many input arguments.")

        start = [None]*2
        if (Narg < self.numargs) or not ('loc' in kwds and
                                         'scale' in kwds):
            # get distribution specific starting locations
            start = self._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.get('loc', start[-2])
        scale = kwds.get('scale', start[-1])
        args += (loc, scale)
        x0, func, restore, args = self._reduce_func(args, kwds)

        optimizer = kwds.get('optimizer', optimize.fmin)
        # convert string to function in scipy.optimize
        if not callable(optimizer) and isinstance(optimizer, string_types):
            if not optimizer.startswith('fmin_'):
                optimizer = "fmin_"+optimizer
            if optimizer == 'fmin_':
                optimizer = 'fmin'
            try:
                optimizer = getattr(optimize, optimizer)
            except AttributeError:
                raise ValueError("%s is not a valid optimizer" % optimizer)
        vals = optimizer(func, x0, args=(ravel(data),), disp=0)
        if restore is not None:
            vals = restore(args, vals)
        vals = tuple(vals)
        return vals
        
class circ_uniform(rv_circular):
    """
		Uniform distribution.
  
           pdf(x) = const = 1/(b-a)
    """
#    def _pdf(self, x):
    def _pdf(self, x):
        return 1/(2*np.pi)
        
    def _cdf(self, x):
        """Интеграл по одному обороту даёт 1, соответственно
        _cdf(x) = x//(2*pi)+x%(2*pi) = x/(2*pi)"""
        return x/(2*np.pi)
        
un = circ_uniform()


class circ_cardioid(rv_circular):
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
    
    def _argcheck(self, rho):
        allpos = np.all(rho > 0)
        return allpos

    def _pdf(self, x, rho):
        return (1 + 2*rho*np.cos(x))/(2*np.pi)    
        
    def _cdf(self, x, rho):
        return (x%(2*np.pi))/(2*np.pi) + rho*(-np.sin(-(x%(2*np.pi))))/(2*np.pi)
        
card = circ_cardioid()
#card1 = circ_cardioid(1, 4)

class circ_von_mises(rv_circular):
    """
		Von Mises distribution (Circular Normal distribution).
          
           defined at p. 35 of 
           "Topics in Circular Statistics"
	      S. R. Jammalamadaka and A. SenGupta.
    """
		
    def _pdf(self, x, kappa):
        return np.exp(kappa*np.cos(x))/(2*np.pi*special.iv(0, kappa))
	

        
mises = circ_von_mises()

class circ_normal(rv_circular):
    """
        Wrapped normal distribution.
        
        Mardia, p.58
    """
    def _pdf(self, x):
        def delta(theta, k):
            return np.exp(-(theta+2*np.pi*k)**2/2)
            
        dens = delta(x, 0)
        i = 1
        it = delta(x, i) + delta(x, -i)
        while it.any() > 2*self.xtol:  
            dens += it
            i += 1
            it = delta(x, i) + delta(x, -i)
        dens /= (2*np.pi)**(1/2)
        return dens

class circ_cauchy(rv_circular):
    """
        Wrapped Cauchy distribution
    """
    
    def _pdf(self, x, a):
        def delta(theta, a, k):
            return 1/(a*a+(theta+2*np.pi*k)**2)
        
        dens = delta(x, a, 0)
        i = 1
        it = delta(x, a, i) + delta(x, a, -i)
        while np.any(it > 2*1e-4): 
            dens += it
            i += 1
            it = delta(x, a, i) + delta(x, a, -i)
        return dens*a/np.pi

class circ_exponential(rv_circular):
    """
        Wrapped exponential distribution
    """
    def _pdf(self, x, l):
        def delta(theta, l, k):
            return l*np.exp(-l*(x+2*np.pi*k))
        
        dens = delta(x, l, 0)
        i = 1
        it = delta(x, l, i)
        while np.any(2*it > 1e-6): 
            dens += it
            i += 1
            it = delta(x, l, i)
        return dens

class circ_levy(rv_circular):
    """
        Wrapped Levy distribution
    """
    def _pdf(self, x, c):
        def delta(theta, c, k):
            return np.exp(-c/(theta+2*np.pi*k))/(theta+2*np.pi*k)**(3/2)
        
        ans = []
        for k in np.nditer([x, c], op_flags=['readwrite']):
            dens = delta(k[0], k[1], 0)
            i = 1
            it1 = delta(k[0], k[1], i) if k[0]+2*np.pi*i > 0 else 0 
            it2 = delta(k[0], k[1], -i) if k[0]-2*np.pi*i > 0 else 0
            while np.any(it1+it2 > 2*1e-4): 
                dens += it1+it2
                i += 1
                it1 = delta(k[0], k[1], i) if k[0]+2*np.pi*i > 0 else 0
                it2 = delta(k[0], k[1], -i) if k[0]-2*np.pi*i > 0 else 0
            ans.append(dens)
        return np.asarray(ans*np.sqrt(c/(2*np.pi)))
#                return x    

normal = circ_normal()
cauchy = circ_cauchy()
expon = circ_exponential()
wlevy = circ_levy()
