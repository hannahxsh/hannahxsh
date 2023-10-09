#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:40:43 2023

@author: xiangsihan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ten_industry = pd.read_excel(r"/Users/xiangsihan/Desktop/QF600/Industry_Portfolios.xlsx")
ten_industry.index = ten_industry["Date"]
ten_industry = ten_industry.drop("Date",axis = 1)

#calculate the mean and standar deviation of ten industries
mean = ten_industry.mean()
std = ten_industry.std()
df_basic = pd.DataFrame(list(mean),list(std)).reset_index()
df_basic.index = mean.index
df_basic.columns = ["std", "mean"]

#calculate covariance
cov = ten_industry.cov()

#Define variables
R = np.array(mean).T
V = np.array(cov)
e = np.ones(len(ten_industry.columns))
V_inv = np.linalg.inv(V)

alpha = np.dot(np.dot(R.T,V_inv), e)
zeta = np.dot(np.dot(R.T,V_inv), R)
delta = np.dot(np.dot(e.T,V_inv), e)

#without riskless asset
Rp = np.linspace(0,2,201)
Rmv = alpha/delta
R_efficient = Rp[Rp>=Rmv]
R_inefficient = Rp[Rp<Rmv]
sigma1 = np.sqrt((delta*R_efficient**2-2*alpha*R_efficient+zeta)/(zeta*delta-alpha**2))
sigma2 = np.sqrt((delta*R_inefficient**2-2*alpha*R_inefficient+zeta)/(zeta*delta-alpha**2))
#efficient frontier
plt.style.use("ggplot")
plt.plot(sigma1, R_efficient, c = "r")
plt.plot(sigma2, R_inefficient, linestyle='--', c = "r")
plt.xlabel("std dev of return")
plt.ylabel("expected return")
plt.title("Efficient Frontier without riskless asset")
plt.show()

#with riskless asset
rf_rate = 0.13
Rp_new = Rp[Rp>=rf_rate]
sigma_riskfree = np.sqrt((Rp_new-rf_rate)**2/(zeta-2*alpha*rf_rate+delta*rf_rate**2))
plt.plot(sigma_riskfree,Rp_new)
plt.xlabel("std dev of return")
plt.ylabel("expected return")
plt.title("Capltal allocation line")
plt.show()

#Asset allocation with riskless asset
R_tg = (alpha*rf_rate-zeta)/(delta*rf_rate-alpha)
Sigma_tg = -np.sqrt(zeta-2*alpha*rf_rate+delta*rf_rate**2)/(delta*(rf_rate-Rmv))

plt.plot(sigma1, R_efficient, c = "#651F13")
plt.plot(sigma2, R_inefficient, linestyle='--', c = "#651F13")
plt.plot(sigma_riskfree,Rp_new, c = "#3A4C4E")
plt.plot(Sigma_tg, R_tg, 'go', label = 'tangency portfolio', c = "#B97A33")
plt.text(Sigma_tg, R_tg, f'({Sigma_tg:.2f}, {R_tg:.2f})', ha='right', va='bottom', fontsize=12, color='black')
plt.xlabel("std dev of return")
plt.ylabel("expected return")
plt.title("Efficient Frontier with riskless asset")
plt.legend(["Efficient Frontier",
    "Inefficient Frontier","CAL","optimal risky portfolio"])

#sharpe ratio
sharpe = (R_tg-rf_rate)/Sigma_tg

#weighted of tangency portfolio
factor = np.dot(zeta,delta)-np.dot(alpha,alpha)
a = (np.dot(np.dot(zeta,V_inv),e)-np.dot(np.dot(alpha,V_inv),R))/factor
b = (np.dot(np.dot(delta,V_inv),R)-np.dot(np.dot(alpha,V_inv),e))/factor
w = a+b*R_tg
w_df = pd.DataFrame(w).reset_index(drop=True)
w_df.index = mean.index

df_basic.to_excel(r'C:\Users\shixi\Desktop\df.xlsx')
w_df.to_excel(r'C:\Users\shixi\Desktop\wdf.xlsx')
