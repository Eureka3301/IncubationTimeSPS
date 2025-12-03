import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import string
from random import randint
import random
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator, LogLocator
import timeit
import os

workdir = 'C:/Grisha/Расчёты различных материалов по SPS/VT6/'
mat_name = 'VT6_UFG_tens'
languages = ['eng']

#************ Material Properties  *********************************
alpha = 1

#************ Read experimental data *******************************

def get_data(data_source):
    f = open(data_source, 'r')
    data1 = np.array([])
    data2 = np.array([])
    name = 'nothing'
    while 1:
        line = f.readline()
        if not line: break
        vals = line.split()
        try:
            data1 = np.append(data1, float(vals[0]))
        except IndexError:
            continue
        except ValueError:
            name = str(vals[0])
            continue
        try:
            data2 = np.append(data2, float(vals[1]))
        except IndexError:
            data1 = np.delete(data1, len(data1) - 1)
        except ValueError:
            data1 = np.delete(data1, len(data1) - 1)
    f.close()

    return (data1, data2, name)


(StrainRateExp, SigmaExp, name) = get_data(workdir + mat_name + '/' + mat_name + '.txt')
StrainRateExp = StrainRateExp[:]
SigmaExp = SigmaExp[:]
N = len(SigmaExp)


try:
    (StrainRateExpScat, SigmaExpScat, name) = get_data(workdir + mat_name + '/' + mat_name + '_scat.txt')
    StrainRateExpScat = StrainRateExpScat[:]
    SigmaExpScat = SigmaExpScat[:]
except FileNotFoundError:
    StrainRateExpScat = np.array([])
    SigmaExpScat = np.array([])

try:
    (StrainRateExpStatic, SigmaExpStatic, name) = get_data(workdir + mat_name + '/' + mat_name + '_static.txt')
    StrainRateExpStatic = StrainRateExpStatic[:]
    SigmaExpStatic = SigmaExpStatic[:]
except FileNotFoundError:
    StrainRateExpStatic = np.array([np.nan])
    SigmaExpStatic = np.array([np.nan])

# Input SPS parameters

try:
    (dat1, dat2, name) = get_data(workdir + mat_name + '/' + mat_name + '_SPSparam.txt')
    (M, q) = (int(dat1[0]), int(dat2[0]))

    # Set interval for search of the tau value
    (low_tau, high_tau) = (dat1[1], dat2[1])

    # Set interval for search of the critical stress
    (low_sigmac, high_sigmac) = (dat1[2], dat2[2])

    (tau_q, sigmac_q) = (int(dat1[3]), int(dat2[3]))
except FileNotFoundError:
    M = 100  # Quantity of random sets (less than 2^N)
    q = 10   # Confidence parameter
    low_tau = 1e-8
    high_tau = 100e-6
    low_sigmac = 50e6
    high_sigmac = 155e6
    tau_q = 500
    sigmac_q = 150

#tau_test_deg = np.linspace(np.log10(low_tau), np.log10(high_tau), tau_q)
#tau_test = pow(10, tau_test_deg)
tau_test = np.linspace(low_tau, high_tau, tau_q)
sigmac_test = np.linspace(low_sigmac, high_sigmac, sigmac_q)
conf = str(int((1 - q / M) * 100))

# Input material properties
(E, sigmaC, mat_name) = get_data(workdir + mat_name + '/' + mat_name + '_prop.txt')

# ************ Main Functions *********************************
# Calculate fracture stress (Sigma*/ SigmaC) dependence on strain rate (epsdot)
# in accordance to incubation time criterion

def Phi(tau, E, alpha, sigmaC, epsdot):
    Xlim = 1e3                  # The limit of fracture time (x = t*/tau) that means times greater are static case ones
    Klim = (Xlim ** (alpha + 1) - (Xlim - 1) ** (alpha + 1)) ** (1 / alpha)
    k = (alpha + 1) ** (1 / alpha) * (sigmaC / (E * epsdot * tau))
    if alpha == 1:                                                                   # analytical solution for alpha = 1
        if k < 1:
            x = k ** 0.5
        else:
            x = (k + 1) / 2
        SigmaStar = (E * epsdot * tau * x) / sigmaC
    else:                                                                  # numerical solution for alpha not equal to 1
         if k < 1:
            x = k ** (alpha / (alpha + 1))
            SigmaStar = (E * epsdot * tau * x) / sigmaC
         else:
            if k <= Klim:
                def func(x):
                    F = (x ** (alpha + 1) - (x - 1) ** (alpha + 1)) ** (1 / alpha) - k
                    return F
                xGuess = (k / (alpha + 1)) ** (1 / alpha)
                if xGuess < 1:
                    xGuess = 1
                x = opt.fsolve(func, xGuess)[0]
                SigmaStar = (E * epsdot * tau * x) / sigmaC
            else:
                SigmaStar = 1
    return SigmaStar

# Derivation function
def NumPrime(dx, f1, f2):
    der = (f2 - f1) / dx
    return der

# Moving average
def moving_average(series, n):
    av_series = np.array([])
    ll = len(series)
    nn = n // 2
    kk = n % 2
    for ii in range(ll):
        if series[ii] != 0:
            if ii < nn:
                av_series = np.append(av_series, np.average(series[:ii + nn + kk]))
            elif nn <= ii <= ll - nn + kk:
                av_series = np.append(av_series, np.average(series[ii - nn:ii + nn - kk]))
            else:
                av_series = np.append(av_series, np.average(series[ii - nn:]))
        else:
            av_series = np.append(av_series, 0)
    return av_series


#***************************** SPS Procedure ********************************************************************
# Generate random signs ****************************

Beta = np.ones([M, N])

# t1 = timeit.default_timer()

# for j in range(M - 1):
#     for i in range(N):
#         if randint(0, 1) > 0.5:
#             Beta[j + 1, i] = 1
#         else:
#             Beta[j + 1, i] = -1

# t2 = timeit.default_timer()
# print(Beta)
# print(t2-t1)

#*********************************************************8

# t1 = timeit.default_timer()

Beta2 = np.ones([M, N])
for j in range(M - 1):
    ff = '{0:0' + str(N) + 'b}'
    signs = ff.format(random.getrandbits(N))
    Beta2[j + 1] = [2 * int(el) - 1 for el in signs]

# t2 = timeit.default_timer()
# print(Beta2)
# print(t2-t1)

# low_tau = 1e-8
# high_tau = 1e-2
# TauQ = 10000

# def SPS_to_plot_conf_interval(sigmaC, Beta, M, q, alpha, E, StrainRateExp, SigmaExp, low_tau, high_tau, TauQ):
#
#     tauPr_deg = np.linspace(np.log10(low_tau), np.log10(high_tau), TauQ)
#     tauPr = pow(10, tauPr_deg)
#
#     SigmaExp_M0 = SigmaExp[:] / sigmaC
#
#     delta_M = np.ones([TauQ, N])
#     for k in range(TauQ):
#         for j in range(N):
#             delta_M[k, j] = Phi(tauPr[k], E, alpha, sigmaC, StrainRateExp[j]) - SigmaExp_M0[j]
#
#     H = abs(delta_M @ Beta.transpose())
#
#     Rank = np.zeros(TauQ)
#     for k in range(TauQ):
#         Rank[k] = len(np.where(H[k] < H[k, 0])[0])
#
#     tau_min = tauPr[np.where(Rank < M - q)[0][0]]
#     tau_max = tauPr[np.where(Rank < M - q)[0][-1]]
#
#     return (tau_min, tau_max)

def SPS_to_plot_conf_interval(sigmaC, Beta, M, q, alpha, E, StrainRateExp, SigmaExp):
    print(r'A new iteration begin: sigmaC={}'.format(sigmaC))

    SigmaExp_M0 = SigmaExp[:] / sigmaC

    low_tau = 1e-10
    high_tau = 1e3
    TauQ = 60
    Rank = np.full(TauQ, M)
    message = 2              #'Ok'
    times_of_mesh_scale = 0

    while len(np.where(Rank <= M - q)[0]) < TauQ - 2 and message == 2:
        tauPr_deg = np.linspace(np.log10(low_tau), np.log10(high_tau), TauQ)
        tauPr = pow(10, tauPr_deg)

        delta_M = np.ones([TauQ, N])
        for k in range(TauQ):
            for j in range(N):
                delta_M[k, j] = Phi(tauPr[k], E, alpha, sigmaC, StrainRateExp[j]) - SigmaExp_M0[j]

        H = abs(delta_M @ Beta.transpose())

        for k in range(TauQ):
            Rank[k] = len(np.where(H[k] < H[k, 0])[0])

        print(Rank)

        if len(np.where(Rank < M - q)[0]) == 0 and times_of_mesh_scale < 3:
            TauQ = TauQ * 2
            Rank = np.full(TauQ, M)
            times_of_mesh_scale += 1
        else:
            if times_of_mesh_scale == 3:
                message = 0                               #'Some of limits is exceeded!'
            else:
                if np.where(Rank < M - q)[0][0] == 0:
                    message = 1                           #'Low limit is exceeded!'
                else:
                    low_tau = tauPr[np.where(Rank < M - q)[0][0] - 1]
                if np.where(Rank < M - q)[0][-1] == TauQ - 1:
                    message = 3                           #'High limit is exceeded!'
                else:
                    high_tau = tauPr[np.where(Rank < M - q)[0][-1] + 1]

    if message == 2:
        tau_min = tauPr[np.where(Rank < M - q)[0][0]]
        tau_max = tauPr[np.where(Rank < M - q)[0][-1]]
    else:
        (tau_min, tau_max) = (1, 2)

    # print(message)

    return (tau_min, tau_max, message)




# Set storage arrays for data

tau_hist = np.zeros(sigmac_q)

taumin = np.zeros(sigmac_q)
taumax = np.zeros(sigmac_q)

message = np.zeros(sigmac_q)

tau_top = tau_hist
tau_bottom = tau_hist


# Constructing of the confidence set

tauSPS = 0
tauSPSmin = 0
tauSPSmax = 0
trmin = high_tau * 1e6
sigmaCmax = high_sigmac
sigmaCmin = 0

# for s in tqdm(range(sigmac_q)):
for s in range(sigmac_q):
    (taumin[s], taumax[s], message[s]) = SPS_to_plot_conf_interval(sigmac_test[s], Beta2, M, q, alpha, E,
                                                                StrainRateExp, SigmaExp)
    tau_hist[s] = (taumax[s] - taumin[s]) / taumin[s]

inlow = np.where(message == 2)[0][0]
inhigh = np.where(message == 2)[0][-1]


tau_hist = tau_hist[inlow:inhigh]
taumax = taumax[inlow:inhigh]
taumin = taumin[inlow:inhigh]

sigmac_test = sigmac_test[inlow:inhigh]
sigmac_res = sigmac_test * 1e-6


tau_hist_av = moving_average(tau_hist, 3)
s = np.where(tau_hist_av == min(tau_hist_av))


tauSPSmin = taumin[s[0][0]]
tauSPSmax = taumax[s[0][0]]
sigmaCSPS = sigmac_test[s[0][0]]
trmin = tau_hist[s[0][0]]

# print(sigmaCSPS, tauSPS)

#************ Data for the second plot of strain-rate dependency on strength *************

(tau1, tau2, message1) = SPS_to_plot_conf_interval(sigmaC[0], Beta2, M, q, alpha, E, StrainRateExp, SigmaExp)

sigmaC1 = sigmaC[0]
sigmaC2 = sigmaC[0]

tau3 = tauSPSmin
sigmaC3 = sigmaCSPS

tau4 = tauSPSmax
sigmaC4 = sigmaCSPS

#*********** Round print values ***********************

if sigmaC1 * 1e-6 > 20:
    sigmaC1_pr = round(sigmaC1 * 1e-6)
    sigmaC3_pr = round(sigmaC3 * 1e-6)
else:
    sigmaC1_pr = round(sigmaC1 * 1e-6, 1)
    sigmaC3_pr = round(sigmaC3 * 1e-6, 1)

if tauSPSmin * 1e6 > 10:
    tauSPSmin_pr = round(tauSPSmin * 1e6)
    tauSPSmax_pr = round(tauSPSmax * 1e6)
else:
    tauSPSmin_pr = round(tauSPSmin * 1e6, 1)
    tauSPSmax_pr = round(tauSPSmax * 1e6, 1)

if tau1 * 1e6 > 10:
    tau1_pr = round(tau1 * 1e6)
    tau2_pr = round(tau2 * 1e6)
else:
    tau1_pr = round(tau1 * 1e6, 1)
    tau2_pr = round(tau2 * 1e6, 1)

# *********** Strain rate interval *******************************************************
NumPoints = 100
plotstatic = 0

if plotstatic == 1:
# if StrainRateExpStatic.any() != np.nan:
    low_rate = 0.5 * min(min(StrainRateExp), min(StrainRateExpStatic))
else:
    low_rate = 0.5 * min(StrainRateExp)

high_rate = 1.5 * max(StrainRateExp)

#low_rate = 2e6
#high_rate = 3e5

strain_rate_deg = np.linspace(np.log10(low_rate), np.log10(high_rate), NumPoints)
strain_rate = pow(10, strain_rate_deg)


# *********** Calculate modeling curves ********************************************************

sigmaD1 = np.zeros(NumPoints)
sigmaD2 = np.zeros(NumPoints)
sigmaD3 = np.zeros(NumPoints)
sigmaD4 = np.zeros(NumPoints)
for i in range(NumPoints):
    sigmaD1[i] = Phi(tau1, E, alpha, sigmaC1, strain_rate[i]) * sigmaC1 * 1e-6
    sigmaD2[i] = Phi(tau2, E, alpha, sigmaC2, strain_rate[i]) * sigmaC2 * 1e-6
    sigmaD3[i] = Phi(tau3, E, alpha, sigmaC3, strain_rate[i]) * sigmaC3 * 1e-6
    sigmaD4[i] = Phi(tau4, E, alpha, sigmaC4, strain_rate[i]) * sigmaC4 * 1e-6


for lang in languages:
    # **************** Plot graph for tau defined above ***********
    text_kwargs = dict(ha='center', va='center', fontsize=16, color='Black')

    # Create Fig and gridspec
    fig1 = plt.figure(figsize=(16.40 / 2.54, 10 / 2.54), dpi=80)
    widths = [2, 3]
    heights = [1.5, 0.5]
    grid = plt.GridSpec(ncols=2, nrows=2, wspace=0.3, hspace=0.08, width_ratios=widths,
                        height_ratios=heights)

    # Define the axes
    ax_main1 = fig1.add_subplot(grid[0, 0], xticklabels=[])
    ax_bottom1 = fig1.add_subplot(grid[1, 0])
    ax_main2 = fig1.add_subplot(grid[0:, 1])
    ax_main2.set_xscale('log')

    if lang != 'rus':
        plot_title = mat_name + ': Confidence is ' + conf + r'%; $\alpha =$' + str(alpha)
        x_label = r'$\sigma_c, MPa$'
        y_label_main1 = r'$\tau, \mu s$'
        y_label_bottom1 = r'$|\tau^{+} - \tau^{-}| / \tau^{-}$'
    else:
        plot_title = 'RB: достоверность ' + conf + r'%;  $\alpha =$' + str(alpha)
        x_label = r'$\sigma_c, МПa$'
        y_label_main1 = r'$\tau, мкс$'
        y_label_bottom1 = r'$|\tau^{+} - \tau^{-}| / \tau^{-}$'

    # ************* Axes main1 **************

    # ax_main1.set_title(plot_title, fontsize=14)
    dx = (sigmac_res[-1] - sigmac_res[0]) / 100
    dy = max(taumax * 1e6) / 100

    ax_main1.set_ylabel(y_label_main1, fontsize=10)
    # ax_main1.yaxis.set_major_locator(LogLocator(base=10, numticks=5))

    #   set axis limits
    ax_main1.set_xlim(sigmac_res[0], sigmac_res[-1])

    # low_tau = 1e-10 * 1e6
    # high_tau = 1e-3 * 1e6
    # ax_main1.set_ylim(low_tau, high_tau)

    ax_main1.tick_params(which='both', axis="both", direction="in", labelsize=8)
    ax_main1.tick_params(which='minor', axis="both", length=0)
    ax_main1.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_main1.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    # ax_main1.scatter(sigmac_res, taumax, s=1.0, marker='o', color='darkturquoise')
    # ax_main1.plot(sigmac_res, taumax * 1e6, color='dimgray')
    # ax_main1.plot(sigmac_res, taumin * 1e6, color='dimgray')
    ax_main1.semilogy(sigmac_res, taumax * 1e6, color='dimgray')
    ax_main1.semilogy(sigmac_res, taumin * 1e6, color='dimgray')
    for s in range(inhigh - inlow):
        ax_main1.vlines(sigmac_res[s], taumin[s] * 1e6, taumax[s] * 1e6, colors='dimgray', linestyles='dotted', lw=1)

    ax_main1.vlines(sigmaCmax * 1e-6, 0, 50, colors='black', linestyles='dotted', lw=1)
    ax_main1.vlines(sigmaCmin * 1e-6, 0, 50, colors='black', linestyles='dotted', lw=1)
    ax_main1.hlines(tauSPSmin * 1e6, sigmaCmin * 1e-6, sigmaCmax * 1e-6, colors='black', linestyles='dotted', lw=1)
    ax_main1.hlines(tauSPSmax * 1e6, sigmaCmin * 1e-6, sigmaCmax * 1e-6, colors='black', linestyles='dotted', lw=1)
    ax_main1.vlines(sigmaCSPS * 1e-6, tauSPSmin * 1e6, tauSPSmax * 1e6, colors='black', linestyles='-', lw=2)
    ax_main1.annotate(r'$\tau = $' + str(tauSPSmin_pr), xy=(sigmaCSPS * 1e-6, tauSPSmin * 1e6),
                      weight='bold', xytext=(1.002 * sigmaCSPS * 1e-6 - 40 * dx, tauSPSmin * 1e6 - 5 * dy),
                      fontsize=6, arrowprops=dict(arrowstyle="->", color='k'))
    ax_main1.annotate(r'$\tau = $' + str(tauSPSmax_pr), xy=(sigmaCSPS * 1e-6, tauSPSmax * 1e6),
                      weight='bold', xytext=(1.002 * sigmaCSPS * 1e-6 + 2 * dx, tauSPSmax * 1e6 + 5 * dy), fontsize=6,
                      arrowprops=dict(arrowstyle="->", color='k'))
    # ax_main1.text(sigmaCmin * 1e-6, tauSPSmin * 1e6, r'$\tau = $' + str(round(tauSPSmin * 1e6, 1)),
    #              fontsize=12, fontweight='bold')
    # ax_main1.text(sigmaCmax * 1e-6, tauSPSmax * 1e6, r'$\tau = $' + str(round(tauSPSmax * 1e6, 1)),
    #              fontsize=12, fontweight='bold')
    # ax_main1.grid(True, which='both')

    # ************* Axes bottom1 **************

    ax_bottom1.set_xlim(sigmac_res[0], sigmac_res[-1])
    # ax_bottom1.set_xlim(low_sigmac, high_sigmac)
    # ax_bottom1.set_ylim(0, max(tau_hist))
    ax_bottom1.set_ylim(0, 5 * min(tau_hist))
    ax_bottom1.tick_params(which='both', axis="both", direction="in", labelsize=8)
    ax_bottom1.tick_params(which='minor', axis="both", length=0)
    ax_bottom1.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_bottom1.yaxis.set_major_locator(MaxNLocator(nbins=3))

    ax_bottom1.set_xlabel(x_label, fontsize=10)
    ax_bottom1.set_ylabel(y_label_bottom1, fontsize=10)
    # ax_bottom1.step(sigmac_test * 1e-6, tau_hist, where='mid', color='deeppink')
    ax_bottom1.step(sigmac_test * 1e-6, tau_hist, where='mid', color='dimgray')
    ax_bottom1.invert_yaxis()
    # ax_bottom1.fill_between(sigmac_test * 1e-6, tau_hist, step="mid", color='deeppink')
    ax_bottom1.fill_between(sigmac_test * 1e-6, tau_hist, step="mid", color='dimgray')
    ax_bottom1.plot(sigmac_test * 1e-6, tau_hist_av, color='black', linestyle=':', lw=1)
    ax_bottom1.annotate(r'$\sigma_{c\_SPS} = $' + str(sigmaC3_pr), xy=(sigmaCSPS * 1e-6, trmin),
                        weight='bold', xytext=(sigmaCSPS * 1e-6 - 1 * dx, 4.75 * min(tau_hist)), fontsize=6,
                        arrowprops=dict(arrowstyle="->", color='k'))
    ax_bottom1.vlines(sigmaCmax * 1e-6, 0, max(tau_hist), colors='black', linestyles='dotted', lw=1)
    ax_bottom1.vlines(sigmaCmin * 1e-6, 0, max(tau_hist), colors='black', linestyles='dotted', lw=1)
    # ax_bottom1.arrow(1.02 * sigmaCSPS * 1e-6, 1.7 * trmin, -0.02 * sigmaCSPS * 1e-6, -0.6 * trmin,
    #                 color='k', ls='-', width=0.1)
    # ax_bottom1.text(1.02 * sigmaCSPS * 1e-6, 2 * trmin, r'$\sigma_c = $' + str(round(sigmaCSPS * 1e-6, 1)),
    #                fontsize=12, weight='bold')
    # ax_bottom1.grid(True)
    # ax_bottom1.vlines(sigmaC2 * 1e-6, 0, 20, colors='k', linestyles='dotted')
    # plot_label = r'$\tau = $' + str(min(tau_hist)) +r'$\mu s$'

    # ************* Axes main2 **************

    if lang != 'rus':
        plot_title = mat_name + ': Confidence is ' + str(int((1 - q / M) * 100)) + r'%; $\alpha =$' + str(alpha)
        ylabel = 'Failure stress, MPa'
        if E == 1:
            xlabel = 'Stress rate, Pa/s'
        else:
            xlabel = 'Strain rate, 1/s'
        Exp_label1 = 'Exp data'
        Exp_label2 = 'Exp data to verify'
        Exp_label3 = 'Steady static exp data'
        SPS_label1 = r'$\sigma_s =$ ' + str(sigmaC1_pr) + ' MPa;\n' \
                     + r'$\tau\ \in [$' + str(tau1_pr) \
                     + '; ' + str(tau2_pr) + r'$] \mu s$'
        SPS_label2 = r'$\sigma_{c\_SPS} = {}$' + str(sigmaC3_pr) + ' MPa;\n' \
                     + r'$\tau\ \in [$' + str(tauSPSmin_pr) \
                     + '; ' + str(tauSPSmax_pr) + r'$] \mu s$'

    else:
        plot_title = 'FB Сжатие: Достоверность ' + str(int((1 - q / M) * 100)) + r'%; $\alpha =$' + str(alpha)
        ylabel = 'Предел прочности, МПа'
        if E == 1:
            xlabel = 'Скорость роста напряжений, Па/с'
        else:
            xlabel = 'Скорость деформации, 1/с'
        Exp_label1 = 'Данные динамических\n испытаний'
        Exp_label2 = 'Данные для проверки'
        Exp_label3 = 'Данные статических\n испытаний'
        SPS_label1 = r'$\sigma_s =$ ' + str(sigmaC1_pr) + ' МПа;\n' \
                     + r'$\tau\ \in [$' + str(tau1_pr) \
                     + '; ' + str(tau2_pr) + '] мкс;'
        SPS_label2 = r'$\sigma_{c\_SPS} = {}$' + str(sigmaC3_pr) + ' МПа;\n' \
                     + r'$\tau\ \in [$' + str(tauSPSmin_pr) \
                     + '; ' + str(tauSPSmax_pr) + '] мкс;'

    # ax_main2.set_title(plot_title, fontsize=14)

    ax_main2.set_xlabel(xlabel, fontsize=8)
    ax_main2.set_ylabel(ylabel, fontsize=8)

    # ax_main2.set_ylim(1000, 2400)

    ax_main2.tick_params(which='both', axis="both", direction="in", labelsize=8)
    ax_main2.tick_params(which='minor', axis="both", length=0)

    ax_main2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_main2.xaxis.set_major_locator(LogLocator(base=10, numticks=5))

    # ax_main2.text(0.1 * max(strain_rate), 0.98 * max(max(sigmaD1), max(sigmaD2), max(sigmaD3), max(sigmaD4)),
    #               mat_name, **text_kwargs)

    ax_main2.plot(strain_rate, sigmaD1, 'k--', label=SPS_label1)
    ax_main2.plot(strain_rate, sigmaD2, 'k--', )
    ax_main2.plot(strain_rate, sigmaD3, 'k-', label=SPS_label2)
    ax_main2.plot(strain_rate, sigmaD4, 'k-')
    if SigmaExpScat.any():
        ax_main2.plot(StrainRateExpScat, SigmaExpScat * 1e-6, 'v', fillstyle='full', markerfacecolor='tab:gray',
                      markeredgecolor='black', label=Exp_label2, markersize=5)
    ax_main2.plot(StrainRateExp, SigmaExp * 1e-6, 'o', fillstyle='full', markerfacecolor='tab:gray',
                  markeredgecolor='black', label=Exp_label1, markersize=5)
    # ax_main2.plot(StrainRateExpStatic, SigmaExpStatic * 1e-6, 's', fillstyle='full', markerfacecolor='tab:gray',
    #               markeredgecolor='black', label=Exp_label3, markersize=5)

    ax_main2.legend(loc=2, markerscale=1.0, fontsize=8)

    plt.tight_layout()
    plt.show()

    # ******************* Dynamic branch in linear scale *********************************

    fig2 = plt.figure(figsize=(10 / 2.54, 7 / 2.54))
    ax2 = fig2.add_subplot(111)

    # ax2.set_title('(a)', fontsize=10, loc='left')

    if lang != 'rus':
        ylabel2 = 'Falure stress, MPa'
        if E == 1:
            xlabel2 = 'Stress rate, Pa/s'
        else:
            xlabel2 = 'Strain rate, 1/s'
        SPS_label1_lin = r'$\sigma_s =$ ' + str(sigmaC1_pr) + ' MPa;\n' \
                         + r'$\tau\ \in [$' + str(tau1_pr) \
                         + '; ' + str(tau2_pr) + r'$] \mu s$'
        SPS_label2_lin = r'$\sigma_{c\_SPS} = {}$' + str(sigmaC3_pr) + ' MPa;\n' \
                         + r'$\tau\ \in [$' + str(tauSPSmin_pr) \
                         + '; ' + str(tauSPSmax_pr) + r'$] \mu s$'
        Exp_label_lin = 'Exp data'

    else:
        ylabel2 = 'Предел прочности, МПа'
        if E == 1:
            xlabel2 = 'Скорость роста напряжений, Па/с'
        else:
            xlabel2 = 'Скорость деформации, 1/с'
        SPS_label1_lin = r'$\sigma_s =$ ' + str(sigmaC1_pr) + ' МПа;\n' \
                         + r'$\tau\ \in [$' + str(tau1_pr) \
                         + '; ' + str(tau2_pr) + '] мкс;'
        SPS_label2_lin = r'$\sigma_{c\_SPS} = {}$' + str(sigmaC3_pr) + ' МПа;\n' \
                         + r'$\tau\ \in [$' + str(tauSPSmin_pr) \
                         + '; ' + str(tauSPSmax_pr) + '] мкс;'
        Exp_label_lin = 'Данные динамических\nиспытаний'

    ax2.set_xlabel(xlabel2, fontsize=8)
    ax2.set_ylabel(ylabel2, fontsize=8)

    ax2.tick_params(which='both', axis="both", direction="in", labelsize=8)
    ax2.tick_params(which='minor', axis="both", length=0)

    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))

    if SigmaExpScat.any():
        RateExp = np.concatenate((StrainRateExp, StrainRateExpScat))
        dx = min(RateExp) * 0.01
        ax2.set_xlim(min(RateExp) - 50 * dx, max(RateExp) + 50 * dx)
        SigExp = np.concatenate((SigmaExp, SigmaExpScat))
        dy = min(SigExp * 1e-6) * 0.01
        ax2.set_ylim(min(SigExp * 1e-6) - 10 * dy, max(SigExp * 1e-6) * 1.5)
    else:
        dx = min(StrainRateExp) * 0.01
        ax2.set_xlim(min(StrainRateExp) - 50 * dx, max(StrainRateExp) + 50 * dx)
        dy = min(SigmaExp * 1e-6) * 0.01
        ax2.set_ylim(min(SigmaExp * 1e-6) - 10 * dy, max(SigmaExp * 1e-6) * 1.5)

    # ax2.text(max(StrainRateExp) + 20 * dx, max(SigmaExp * 1e-6) * 1.5 - 20 * dy, mat_name, **text_kwargs)

    # Color version

    # ax2.plot(strain_rate, sigmaD1, 'r--', label=SPS_label1_lin, linewidth=1)
    # ax2.plot(strain_rate, sigmaD2, 'r--', linewidth=1)
    # ax2.plot(strain_rate, sigmaD4, 'g-', label=SPS_label2_lin, linewidth=1)
    # ax2.plot(strain_rate, sigmaD3, 'g-', linewidth=1)
    # ax2.plot(StrainRateExp, SigmaExp * 1e-6, 'bo', label=Exp_label1, markersize=5)

    # Gray scale version

    ax2.plot(strain_rate, sigmaD1, 'k--', label=SPS_label1_lin, linewidth=1)
    ax2.plot(strain_rate, sigmaD2, 'k--', linewidth=1)
    ax2.plot(strain_rate, sigmaD4, 'k-', label=SPS_label2_lin, linewidth=1)
    ax2.plot(strain_rate, sigmaD3, 'k-', linewidth=1)
    ax2.plot(StrainRateExp, SigmaExp * 1e-6, 'o', fillstyle='full', markerfacecolor='tab:gray',
             markeredgecolor='black', label=Exp_label_lin, markersize=5)
    if SigmaExpScat.any():
        ax2.plot(StrainRateExpScat, SigmaExpScat * 1e-6, 'v', fillstyle='full', markerfacecolor='tab:gray',
                 markeredgecolor='black', label=Exp_label2, markersize=5)

    ax2.legend(loc=2, markerscale=1.0, fontsize=8, framealpha=0.7)

    plt.tight_layout()
    plt.show()

    # ******************* Confidence set *********************************

    fig3 = plt.figure(figsize=(8.20 / 2.54, 7 / 2.54))
    ax3 = fig3.add_subplot(111)

    dx = (sigmac_res[-1] - sigmac_res[0]) / 100
    dy = max(taumax * 1e6) / 100

    x_label3 = r'$\sigma_c, MPa$'
    y_label3 = r'$\tau, \mu s$'

    ax3.set_ylabel(y_label3, fontsize=12)
    ax3.set_xlabel(x_label3, fontsize=12)
    ax3.set_xlim(sigmac_res[0], sigmac_res[-1])
    # ax_main1.set_xlim(low_sigmac, high_sigmac)
    # ax3.set_ylim(0, max(taumax * 1e6))
    ax3.tick_params(which='both', axis="both", direction="in", labelsize=8)
    ax3.tick_params(which='minor', axis="both", length=0)

    # ***** Hide labels ************************************

    # labels = [item.get_text() for item in ax3.get_xticklabels()]
    # empty_string_labels = ['']*len(labels)
    # ax3.set_xticklabels(empty_string_labels)
    # labels = [item.get_text() for item in ax3.get_yticklabels()]
    # empty_string_labels = ['']*len(labels)
    # ax3.set_yticklabels(empty_string_labels)

    # ***************************************************************

    ax3.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax3.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    # ax3.plot(sigmac_res, taumax * 1e6, color='dimgray')
    # ax3.plot(sigmac_res, taumin * 1e6, color='dimgray')
    ax3.semilogy(sigmac_res, taumax * 1e6, color='dimgray')
    ax3.semilogy(sigmac_res, taumin * 1e6, color='dimgray')
    for s in range(inhigh - inlow):
        ax3.vlines(sigmac_res[s], taumin[s] * 1e6, taumax[s] * 1e6, colors='dimgray', linestyles='dotted', lw=1)
    # ax3.vlines(sigmaCmax * 1e-6, 0, 50, colors='dimgray', linestyles='dotted', lw=1)
    # ax3.vlines(sigmaCmin * 1e-6, 0, 50, colors='dimgray', linestyles='dotted', lw=1)
    ax3.hlines(tauSPSmin * 1e6, sigmac_test[0] * 1e-6, sigmac_test[-1] * 1e-6, colors='black', linestyles='dotted',
               lw=1.5)
    ax3.hlines(tauSPSmax * 1e6, sigmac_test[0] * 1e-6, sigmac_test[-1] * 1e-6, colors='black', linestyles='dotted',
               lw=1.5)
    ax3.annotate(r'$\tau = $' + str(tauSPSmin_pr), xy=(sigmaCSPS * 1e-6, tauSPSmin * 1e6), weight='bold',
                 xytext=(1.002 * sigmaCSPS * 1e-6 - 40 * dx, tauSPSmin * 1e6 - 5 * dy), fontsize=6,
                 arrowprops=dict(arrowstyle="->", color='k'))
    ax3.annotate(r'$\tau = $' + str(tauSPSmax_pr), xy=(sigmaCSPS * 1e-6, tauSPSmax * 1e6), weight='bold',
                 xytext=(1.002 * sigmaCSPS * 1e-6 + 10 * dx, tauSPSmax * 1e6 + 5 * dy), fontsize=6,
                 arrowprops=dict(arrowstyle="->", color='k'))
    ax3.vlines(sigmaCSPS * 1e-6, tauSPSmin * 1e6, tauSPSmax * 1e6, colors='black', linestyles='-', lw=2)
    ax3.vlines(sigmaCSPS * 1e-6, 0, tau_test[-1] * 1e6, colors='black', linestyles='dotted', lw=1.5)
    ax3.annotate(r'$\sigma_{c\_SPS} = $' + str(sigmaC3_pr), xy=(sigmaCSPS * 1e-6, 0), weight='bold',
                 xytext=(1.002 * sigmaCSPS * 1e-6 - 50 * dx, 7 * dy), fontsize=6,
                 arrowprops=dict(arrowstyle="->", color='k'))

    plt.tight_layout()
    plt.show()

    # ************ save data to files ****************************************************

    try:
        os.makedirs(workdir + mat_name + '_res_2d_tau_sigmaC/')
        file_name1 = workdir + mat_name + '_res_2d_tau_sigmaC/' + mat_name + 'tau_sigmac_' + conf + lang + '.png'
        file_name2 = workdir + mat_name + '_res_2d_tau_sigmaC/' + mat_name + 'tau_sigmac_' + conf + lang + '_linear.png'
        file_name3 = workdir + mat_name + '_res_2d_tau_sigmaC/' + mat_name + 'tau_sigmac_' + conf + lang + '_set.png'
        file_name_data = workdir + mat_name + '_res_2d_tau_sigmaC/' + mat_name + 'tau_sigmac_' + conf + lang + '.txt'
    except OSError:
        file_name1 = workdir + mat_name + '_res_2d_tau_sigmaC/' + mat_name + 'tau_sigmac_' + conf + lang + '.png'
        file_name2 = workdir + mat_name + '_res_2d_tau_sigmaC/' + mat_name + 'tau_sigmac_' + conf + lang + '_linear.png'
        file_name3 = workdir + mat_name + '_res_2d_tau_sigmaC/' + mat_name + 'tau_sigmac_' + conf + lang + '_set.png'
        file_name_data = workdir + mat_name + '_res_2d_tau_sigmaC/' + mat_name + 'tau_sigmac_' + conf + lang + '.txt'

    fig1.savefig(file_name1, dpi=300)
    fig2.savefig(file_name2, dpi=300)
    fig3.savefig(file_name3, dpi=300)






