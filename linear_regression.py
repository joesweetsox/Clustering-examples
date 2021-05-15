# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 18:51:44 2021

@author: patri
"""
import numpy as np

nobs=1000000
sigma=.5
x=np.linspace(-5,3.5,nobs)
Ey=2.3*x**3+5.6*x**2-4*x+1
xy=np.array([x,Ey]).T
tots=np.ones(xy.shape[0]).T@xy

means=np.ones(xy.shape[0]).T@xy/xy.shape[0]

#linear regression example

#define the design matrix to include a constant
x_linear_design=np.array([np.ones(nobs),x]).T

#add in some error to y
e=np.random.normal(loc=0,scale=sigma,size=nobs)

y=Ey+e

#estimate beta

linear_beta_hat=np.linalg.inv(x_linear_design.T@x_linear_design)@x_linear_design.T@y

#but the form might be quadratic, create a new design matrix
x_quadratic_design=np.array([np.ones(nobs),x,x**2]).T
quadratic_beta_hat=np.linalg.inv(x_quadratic_design.T@x_quadratic_design)@x_quadratic_design.T@y

#check for cubic
x_cubic_design=np.array([np.ones(nobs),x,x**2,x**3]).T
cubic_beta_hat=np.linalg.inv(x_cubic_design.T@x_cubic_design)@x_cubic_design.T@y

linear_fitted=x_linear_design@linear_beta_hat

linear_error=y-linear_fitted

lin_beta_error=linear_error.T@linear_error

sse_lin_error=linear_error.T@linear_error


quadratic_fitted=x_quadratic_design@quadratic_beta_hat

quad_error=y-quadratic_fitted

quad_beta_error=quad_error.T@quad_error

sse_quad_error=quad_error.T@quad_error

cubic_fitted=x_cubic_design@cubic_beta_hat

cubic_error=y-cubic_fitted

cubic_beta_error=cubic_error.T@cubic_error

sse_cubic_error=cubic_error.T@cubic_error


print(sse_lin_error, sse_quad_error, sse_cubic_error, sse_lin_error/sse_quad_error,sse_lin_error/sse_cubic_error)

print(linear_beta_hat,"\n",quadratic_beta_hat,"\n",cubic_beta_hat)

#do some plots
import matplotlib.pyplot as plt
plt.figure(dpi=600)
plt.plot(x,y,'-b',linestyle='',marker='o',markersize=1,label='data')
plt.plot(x,linear_fitted,'-g',label='linear fit')
plt.plot(x,quadratic_fitted,'-r',label='quadratic fit')
plt.plot(x,cubic_fitted,'-y',label='cubic fit')
plt.legend()

#diagnostic plots
plt.figure(dpi=600)
plt.subplot(211)
plt.plot(x,linear_error,linestyle='',marker='.',markersize=1)
plt.subplot(212)
plt.plot(np.sort(linear_error),np.sort(np.random.normal(loc=0,scale=sigma,size=nobs)),linestyle='',marker='.',markersize=1)

plt.figure(dpi=600)
plt.subplot(211)
plt.plot(x,quad_error,linestyle='',marker='.',markersize=1)
plt.subplot(212)
plt.plot(np.sort(quad_error),np.sort(np.random.normal(loc=0,scale=sigma,size=nobs)),linestyle='',marker='.',markersize=1)

plt.figure(dpi=600)
plt.subplot(211)
plt.plot(x,cubic_error,linestyle='',marker='.',markersize=1)
plt.subplot(212)
plt.plot(np.sort(cubic_error),np.sort(np.random.normal(loc=0,scale=sigma,size=nobs)),linestyle='',marker='.',markersize=1)
