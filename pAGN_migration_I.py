#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:24:00 2024

@author: olga
"""

from pagn import Thompson
from pagn import Sirko
import numpy as np
import matplotlib.pyplot as plt
import pagn.constants as ct
plt.rcParams['figure.dpi'] = 150
from IPython.display import clear_output
from scipy.interpolate import UnivariateSpline
from pagn.opacities import electron_scattering_opacity
import matplotlib.lines as mlines
#%%
def gamma_0(q, hr, Sigma, r, Omega):
    """
    Method to find the normalization torque

    Parameters
    ----------
    q: float/array
        Float or array representing the mass ratio between the migrator and the central BH.
    hr: float/array
        Float or array representing the disk height to distance from central BH ratio.
    Sigma: float/array
        Float or array representing the disk surface density in kg/m^2
    r: float/array
        Float or array representing the distance from the central BH in m
    Omega: float/array
        Float or array representing the angular velocity at the migrator position in SI units.

    Returns
    -------
    gamma_0: float/array
        Float or array representing the single-arm migration torque on the migrator in kg m^2/ s^2.

    """
    gamma_0 = q*q*Sigma*r*r*r*r*Omega*Omega/(hr*hr)
    return gamma_0


def gamma_iso(dSigmadR, dTdR):
    """
    Method to find the locally isothermal torque.

    Parameters
    ----------
    dSigmadR: float/array
        Discrete array representing the log surface density gradient in the disk.
    dTdR: float/array
        Discrete array representing the log thermal gradient in the disk.

    Returns
    -------
    gamma_iso: float/array
        Float or array representing the locally isothermal torque on the migrator in kg m^2/ s^2.

    """
    alpha = - dSigmadR
    beta = - dTdR
    gamma_iso = - 0.85 - alpha - 0.9*beta
    return gamma_iso


def gamma_ad(dSigmadR, dTdR):
    """
    Method to find the adiabatic torque.

    Parameters
    ----------
    dSigmadR: float/array
        Discrete array representing the log surface density gradient in the disk.
    dTdR: float/array
        Discrete array representing the log thermal gradient in the disk.

    Returns
    -------
    gamma_ad: float/array
        Float or array representing the adabiatic torque on the migrator in kg m^2/ s^2.

    """
    alpha = - dSigmadR
    beta = - dTdR
    gamma = 5/3
    xi = beta - (gamma - 1)*alpha
    gamma_ad = - 0.85 - alpha - 1.7*beta + 7.9*xi/gamma
    return gamma_ad


def dSigmadR(obj):
    """
    Method that interpolates the surface density gradient of an AGN disk object.

    Parameters
    ----------
    obj: object
        Either a SirkoAGN or ThompsonAGN object representing the AGN disk being considered.

    Returns
    -------
    dSigmadR: float/array
        Discrete array of the log surface density gradient.

    """
    Sigma = 2*obj.rho*obj.h # descrete
    rlog10 = np.log10(obj.R)  # descrete
    Sigmalog10 = np.log10(Sigma)  # descrete
    Sigmalog10_spline = UnivariateSpline(rlog10, Sigmalog10, k=3, s=0.005, ext=0)  # need scipy ver 1.10.0
    dSigmadR_spline =  Sigmalog10_spline.derivative()
    dSigmadR = dSigmadR_spline(rlog10)
    return dSigmadR


def dTdR(obj):
    """
    Method that interpolates the thermal gradient of an AGN disk object.

    Parameters
    ----------
    obj: object
        Either a SirkoAGN or ThompsonAGN object representing the AGN disk being considered.

    Returns
    -------
    dTdR: float/array
        Discrete array of the log thermal gradient.

    """
    rlog10 = np.log10(obj.R)  # descrete
    Tlog10 = np.log10(obj.T)  # descrete
    Tlog10_spline = UnivariateSpline(rlog10, Tlog10, k=3, s=0.005, ext=0)  # need scipy ver 1.10.0
    dTdR_spline = Tlog10_spline.derivative()
    dTdR = dTdR_spline(rlog10)
    return dTdR


def dPdR(obj):
    """
    Method that interpolates the total pressure gradient of an AGN disk object.

    Parameters
    ----------
    obj: object
        Either a SirkoAGN or ThompsonAGN object representing the AGN disk being considered.

    Returns
    -------
    dPdR: float/array
        Discrete array of the log total pressure gradient.

    """
    rlog10 = np.log10(obj.R)  # descrete
    pgas = obj.rho * obj.T * ct.Kb / ct.massU
    prad = obj.tauV*ct.sigmaSB*obj.Teff4/(2*ct.c)
    ptot = pgas + prad
    Plog10 = np.log10(ptot)  # descrete
    Plog10_spline = UnivariateSpline(rlog10, Plog10, k=3, s=0.005, ext=0)  # need scipy ver 1.10.0
    dPdR_spline = Plog10_spline.derivative()
    dPdR = dPdR_spline(rlog10)
    return dPdR


def CI_p10(dSigmadR, dTdR):
    """
    Method to calculate torque coefficient for the Paardekooper et al. 2010 values.

    Parameters
    ----------
    dSigmadR: float/array
        Discrete array representing the log surface density gradient in the disk.
    dTdR: float/array
        Discrete array representing the log thermal gradient in the disk.

    Returns
    -------
    cI: float/array
        Paardekooper et al. 2010 migration torque coefficient
    """
    cI = -0.85 + 0.9*dTdR + dSigmadR
    return cI


def CI_jm17_tot(dSigmadR, dTdR, gamma, obj):
    """
    Method to calculate torque coefficient for the Jiménez and Masset 2017 values.

    Parameters
    ----------
    dSigmadR: float/array
        Discrete array representing the log surface density gradient in the disk.
    dTdR: float/array
        Discrete array representing the log thermal gradient in the disk.
    gamma: float
        Adiabatic index
    obj: object
        Either a SirkoAGN or ThompsonAGN object representing the AGN disk being considered.


    Returns
    -------
    cI: float/array
        Jiménez and Masset 2017 migration torque coefficient
    """
    cL = CL(dSigmadR, dTdR, gamma, obj)
    cI = cL + (0.46 + 0.96*dSigmadR - 1.8*dTdR)/gamma
    return cI


def CI_jm17_iso(dSigmadR, dTdR):
    """
    Method to calculate the locally isothermal torque coefficient for the Jiménez and Masset 2017 values.

    Parameters
    ----------
    dSigmadR: float/array
        Discrete array representing the log surface density gradient in the disk.
    dTdR: float/array
        Discrete array representing the log thermal gradient in the disk.

    Returns
    -------
    cI: float/array
        Jiménez and Masset 2017 migration locally isothermal torque coefficient
    """
    cI = -1.36 + 0.54*dSigmadR + 0.5*dTdR
    return cI


def CL(dSigmadR, dTdR, gamma, obj):
    """
    Method to calculate the Lindlblad torque for the Jiménez and Masset 2017 values.

    Parameters
    ----------
    dSigmadR: float/array
        Discrete array representing the log surface density gradient in the disk.
    dTdR: float/array
        Discrete array representing the log thermal gradient in the disk.
    gamma: float
        Adiabatic index
    obj: object
        Either a SirkoAGN or ThompsonAGN object representing the AGN disk being considered.


    Returns
    -------
    cL: float/array
        Jiménez and Masset 2017 Lindblad torque coefficient
    """
    xi = 16*gamma*(gamma - 1)*ct.sigmaSB*(obj.T*obj.T*obj.T*obj.T)\
         /(3*obj.kappa*obj.rho*obj.rho*obj.h*obj.h*obj.Omega*obj.Omega)
    x2_sqrt = np.sqrt(xi/(2*obj.h*obj.h*obj.Omega))
    fgamma = (x2_sqrt + 1/gamma)/(x2_sqrt+1)
    cL = (-2.34 - 0.1*dSigmadR + 1.5*dTdR)*fgamma
    return cL


def gamma_thermal(gamma, obj, q):
    """
    Method to calculate the thermal torque from the Masset 2017 equations, with decay and torque saturation.

    Parameters
    ----------
    gamma: float
        Adiabatic index
    obj: object
        Either a SirkoAGN or ThompsonAGN object representing the AGN disk being considered.
    q: float/array
        Float or array representing the mass ratio between the migrator and the central BH.

    Returns
    -------
    g_thermal: float/array
        Masset 2017 migration total thermal torque.
    """
    xi = 16 * gamma * (gamma - 1) * ct.sigmaSB * (obj.T * obj.T * obj.T * obj.T) \
         / (3 * obj.kappa * obj.rho * obj.rho * obj.h * obj.h * obj.Omega * obj.Omega)
    mbh = obj.Mbh*q
    muth = xi * obj.cs / (ct.G * mbh)
    R_Bhalf = ct.G*mbh/obj.cs**2
    muth[obj.h<R_Bhalf] = (xi / (obj.cs*obj.h))[obj.h<R_Bhalf]

    Lc = 4*np.pi*ct.G*mbh*obj.rho*xi/gamma
    lam = np.sqrt(2*xi/(3*gamma*obj.Omega))

    dP = -dPdR(obj)
    xc = dP*obj.h*obj.h/(3*gamma*obj.R)

    kes = electron_scattering_opacity(X=0.7)
    L = 4 * np.pi * ct.G * ct.c * mbh / kes

    g_hot = 1.61*(gamma - 1)*xc*L/(Lc*gamma*lam)
    g_cold = -1.61*(gamma - 1)*xc/(gamma*lam)
    g_thermal = g_hot + g_cold
    g_thermal_new = g_hot*(4*muth/(1+4*muth)) + g_cold*(2*muth/(1+2*muth))
    g_thermal[muth < 1] = g_thermal_new[muth < 1]
    decay = 1 - np.exp(-lam*obj.tauV/obj.h)
    return g_thermal*decay

#%%

from IPython.display import clear_output

disk_name = ['sirko', 'thompson']
d_counter = 0

f, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey='row', gridspec_kw=dict(hspace=0, wspace =0, height_ratios = (2, 2, 2, 1.2)), tight_layout=True)
for axx in axes.flatten():
    axx.set_yscale('log')
    axx.set_xscale('log')

for dname in disk_name:
    Mbh = 1e6
    q = 5e-6

    #generate the disk values for both AGN disk models using pagn
    if dname == 'thompson':
        objin = Thompson.ThompsonAGN(Mbh = Mbh*ct.MSun, Mdot_out=0.,)
        rout = objin.Rs*(1e7)
        sigma = 200 * (Mbh / 1.3e8) ** (1 / 4.24)
        Mdot_out = 1.5e-2
        obj = Thompson.ThompsonAGN(Mbh=Mbh*ct.MSun, Rout = rout, Mdot_out=Mdot_out*ct.MSun/ct.yr)
        clear_output(obj.solve_disk(N=1e4))
    else:
        le = 0.5
        alpha = 0.01
        obj = Sirko.SirkoAGN(Mbh=Mbh*ct.MSun, le=le, alpha=alpha, b=0)
        clear_output(obj.solve_disk(N=1e4))

    Gamma_0 = gamma_0(q, obj.h / obj.R, 2 * obj.rho * obj.h, obj.R, obj.Omega)

    #Grishin et al 2023 equations
    dSig = dSigmadR(obj)
    dT = dTdR(obj)
    cI_p10 = CI_p10(dSig, dT)
    Gamma_I_p10 = cI_p10*Gamma_0
    gamma = 5/3

    cI_jm_tot = CI_jm17_tot(dSig, dT, gamma, obj)
    Gamma_I_jm_tot = cI_jm_tot*Gamma_0
    Gamma_therm = gamma_thermal(gamma, obj, q)*Gamma_0*obj.R/obj.h

    Gamma_tot = Gamma_therm + Gamma_I_jm_tot

    #-----Plotting-----#


    linestyles = ['-', '--', '-.', ':']
    ax = axes[:, d_counter]
    if hasattr(obj, 'alpha'):
        ax[0].text(10 ** 1.2, 10 ** 40,  r'${\rm Sirko \, and \, Goodman}$' )
    else:
        ax[0].text(10 ** 1.2, 10 ** 40,  r'${\rm Thompson}$')

    for iGamma, Gamma in enumerate([Gamma_I_jm_tot, Gamma_therm, Gamma_tot]):
        maskg = Gamma >= 0
        indices = np.nonzero(maskg[1:] != maskg[:-1])[0] + 1
        Gammas = np.split(Gamma, indices)
        Rs = np.split(obj.R, indices)
        ignnum = 0
        ignum2 = 0
        for iseg, seg in enumerate(Gammas):
            if seg[0] < 0.:
                if Rs[iseg][0] / obj.Rs > ignnum + 40:
                    ax[iGamma].axvline(Rs[iseg][0] / obj.Rs, -100, 100, color = 'k', alpha = 0.1)
                    ignnum = Rs[iseg][0] / obj.Rs

                ax[iGamma].plot(Rs[iseg]/obj.Rs, abs(seg)*ct.SI_to_gcm2, c='C1', zorder = 2)
                if iGamma == 2 and Rs[iseg][0] / obj.Rs > ignum2 + 40:
                    ax[3].axvline(Rs[iseg][0] / obj.Rs, -100, 100, color='k', alpha=0.1)
                    ignum2 = Rs[iseg][0] / obj.Rs

            else:
                ax[iGamma].plot(Rs[iseg] / obj.Rs, abs(seg*ct.SI_to_gcm2) , c='C0', zorder = 2)
        if iGamma == 0:
            Gamma2 = Gamma_I_p10
            maskg2 = Gamma2 >= 0
            indices2 = np.nonzero(maskg2[1:] != maskg2[:-1])[0] + 1
            Gammas2 = np.split(Gamma2, indices2)
            Rs2 = np.split(obj.R, indices2)
            for iseg2, seg2 in enumerate(Gammas2):
                if seg2[0] < 0.:
                    ax[iGamma].plot(Rs2[iseg2] / obj.Rs, abs(seg2), c='C1', zorder = 1, alpha = 0.4)

                else:
                    ax[iGamma].plot(Rs2[iseg2] / obj.Rs, abs(seg2), c='C0', zorder = 1, alpha = 0.4)
    ax[3].plot(obj.R/obj.Rs, 2*obj.h*obj.rho*ct.SI_to_gcm2, label = r"$\Sigma_{\rm g} [{\rm g cm}^{-2}]$")
    d_counter += 1

pos_line = mlines.Line2D([], [], color='C0', marker='s',
                           markersize=0, label=r'$\rm{Outward}$')
neg_line = mlines.Line2D([], [], color='C1', marker='s',
                            markersize=0, label=r'$\rm{Inward}$')
artists_handles = [pos_line, neg_line]
axes[2, 1].legend(handles=artists_handles, framealpha = 1)

pos_line2 = mlines.Line2D([], [], color='C0', marker='s', alpha = 0.4,
                         markersize=0,)
neg_line2 = mlines.Line2D([], [], color='C1', marker='s', alpha = 0.4,
                         markersize=0,)

from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
axes[0,1].legend(handles=[(pos_line, neg_line), (pos_line2, neg_line2,) ],
                 labels=[r'${\rm Jim \acute{e} nez \, and \, Masset \, (2017)}$', r'$\rm Paardekooper \, et \, al. \, (2010)$',],
                 handler_map = {tuple: HandlerTuple(ndivide = None, pad = 0.)},
                 framealpha = 1)

axes[0,0].set_ylabel(r'${\Gamma_{\rm I} \, {\rm [g \, cm}^{2}{\rm s}^{-2}{\rm ]} }$')
axes[1,0].set_ylabel(r'${\Gamma_{\rm therm} \, {\rm [g \, cm}^{2}{\rm s}^{-2}{\rm ]} }$')
axes[2,0].set_ylabel(r'${\Gamma_{\rm tot} \, {\rm [g \, cm}^{2}{\rm s}^{-2}{\rm ]} }$')
axes[3, 0].set_ylabel(r'$\Sigma_{\rm g} [{\rm g \, cm}^{-2}]$')

x_label = r"$r \, [R_{\rm s}]$"
axes[3, 0].set_xlabel(x_label)
axes[3, 1].set_xlabel(x_label)

axes[0, 0].set_ylim((1e25, 1e42))


axes[1, 0].set_ylim((5e24, 1e42))

axes[2, 0].set_ylim((1e25, 1e42))


axes[3, 0].set_ylim((1e1, 1e7))

for axx in axes.flatten():
    axx.yaxis.set_ticks_position('both')
    axx.xaxis.set_ticks_position('both')
    axx.set_xlim((1e1, 1e7))
axes[0,1].set_xlim((1.1e1, 1e7))

f.align_ylabels()
plt.show()