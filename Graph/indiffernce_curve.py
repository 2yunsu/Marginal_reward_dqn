"""
Code from :
https://deepnote.com/app/gaby-galvan/Edgeworth-Box-5c892517-cb51-4f3f-b20d-8e5dd8b5ac46
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

out = 0
draw_general_curve = False

#The main function 
def draw_Ed_Bow(U1,U2, Xmax, Ymax, Xmin=10**(-6), Ymin=10**(-6), Utility_Show = False, Num_ind = 10,Xlab ="X",Ylab="Y",e=200, Contract_draw=True,AlPoint = None,colors =["black","Orange", "blue","red"],Utility_draw = True,marginal_rate = None, al_color="black"):
    """
    Input : 
        U1: Utility of the 1st agent (must depend on 2 variables)
        U2: Utility of the 2nd agent (must depend on 2 variables)
        Xmax : the limit of the box
        Ymax: the limit of the box
        Xmin=10**(-6): the limit of the box (default: set to ~0 to avoid problems with log expressions)
        Ymin=10**(-6): the box limit (default: set to ~0 to avoid problems with log expressions)
        Utility_Show = False: Show utility levels on the Edgeworth box
        Num_ind = 10 : Number of indifference curves per agent
        e = 200 : Number of steps to compute the utility levels and the contract curve
        Contract_draw = True: Draw the contract curve 
        AlPoint = None : show an allocation point and his 2 indifference curves. Should be a tuple/list (x,y)
        colors = ["black","Orange", "blue","red"] : To choose the color of : [ContractCurve, endowment point, indifference curves Agent1, indifference curves agent2]
        Utility_draw = True : Draw the indefrence curves
    Output:
        None (but draws the Edgeworth box) 
    """
    delta = min((Xmax-Xmin)/e,(Ymax-Ymin)/e)

    x = np.arange(Xmin, Xmax, delta) #Tomates
    y = np.arange(Ymin, Ymax, delta) #courgettes
    X, Y = np.meshgrid(x, y)
    Z1 = lambda x,y : U1(x,y)
    Z2 = lambda x,y : U2(Xmax-x,Ymax-y)

    #the contract curve
    Num_ind_1 = Num_ind
    Num_ind_2 = Num_ind

    if Contract_draw == True:
        Z2grad = np.gradient(Z2(X,Y))
        Z1grad = np.gradient(Z1(X,Y))

        global out
        out = (Z2grad[0]*Z1grad[1]-Z2grad[1]*Z1grad[0])

        Cont = plt.contour(X,Y,out,colors=colors[0],levels=[0])
        fmt = {}
        strs = ["Contract curve"]
        for l, s in zip(Cont.levels, strs):
            fmt[l] = s
        plt.clabel(Cont, Cont.levels, inline = True,
                fmt = fmt, fontsize = 10)
        
        C_curv = abs(pd.DataFrame(out ,index=y, columns=x))
        C_curv = C_curv.index @ (C_curv == C_curv.apply(min))
        xC_curv = np.arange(Xmin,Xmax,(Xmax-Xmin)/(Num_ind+1))
        C_curv = np.interp(xC_curv,C_curv.index,C_curv)
        Num_ind_1 = pd.Series(Z1(xC_curv,C_curv)).sort_values(ascending=True)
        Num_ind_2 = pd.Series(Z2(xC_curv,C_curv)).sort_values(ascending=True)
        
    #Draw the dotation point and his curves
    if AlPoint != None:
        if marginal_rate:
            plt.scatter(AlPoint[0],AlPoint[1],s=200,marker=".",color = al_color,label=marginal_rate)
            # plt.text(AlPoint[0],AlPoint[1],marginal_rate, fontsize=10, color="black", verticalalignment='center', horizontalalignment='center')
        else:
            plt.scatter(AlPoint[0],AlPoint[1],s=200,marker=".",color = colors[1],label="Allocation point")
        Num_ind_1 = [Z1(AlPoint[0],AlPoint[1])]
        Num_ind_2 = [Z2(AlPoint[0],AlPoint[1])]

    #draw the indifference curve
    if Utility_draw == True:
        C1 = plt.contour(X, Y, Z1(X,Y),colors = colors[2],levels=Num_ind_1)
        C2 = plt.contour(X, Y, Z2(X,Y),colors = colors[3],levels=Num_ind_2)
        if Utility_Show == True:
            fmt = {}
            strs = round(pd.Series(C1.levels[:]),1)
            for l, s in zip(C1.levels, strs):
                fmt[l] = s
            # breakpoint()
            plt.clabel(C1, C1.levels, inline = True,
                    fmt = fmt, fontsize = 10)
            #Utility level2

            fmt = {}
            strs = round(pd.Series(C2.levels[:]),1)

            for l, s in zip(C2.levels, strs):
                fmt[l] = s
            plt.clabel(C2, C2.levels, inline = True,
                    fmt = fmt, fontsize = 10)

    if draw_general_curve == True:
        plt.title("Indifference Curves")
    else:
        plt.title("Experiment's Result")
    offset = 0.1  # 바깥쪽으로 이동할 거리
    plt.tick_params(axis="both", direction="in", top=True, right=True)
    plt.text(Xmin - offset, Ymin - offset, r"$A_1$", fontsize=20, color="blue", verticalalignment='top', horizontalalignment='right')
    plt.text(Xmax + offset, Ymax + offset, r"$A_2$", fontsize=20, color="red", verticalalignment='bottom', horizontalalignment='left')
    plt.xlabel(Xlab)
    plt.ylabel(Ylab)

#Utility of 1st agent (depend on X,Y):
# U1 = lambda c,n : 10*c**0.8 * n**0.8
U1 = lambda c, n: 10 * (1 - (0.8)**c) / (1 - 0.8) + 10* (1 - (0.8)**n) / (1 - 0.8)
#Utility of 2nd agent (depend on X,Y):
# U2 = lambda c,n : 10*c**0.8 * n**0.8
U2 = lambda c, n: 10 * (1 - (0.8)**c) / (1 - 0.8) + 10* (1 - (0.8)**n) / (1 - 0.8)

if draw_general_curve: # #draw the general indifference curve
    draw_Ed_Bow(U1,U2,6,6,colors=["k","Orange", "lightblue","mistyrose"],Utility_Show = True, Num_ind=5)
    draw_Ed_Bow(U1,U2,6,6,Xlab=r"$X^1$",Ylab=r"$X^2$",AlPoint=(3,3),Contract_draw=True,Utility_Show = True)
    plt.savefig("./indiffernce_curve.png")

else: #draw the experiment result indifference curve
    draw_Ed_Bow(U1,U2,6,6,colors=["k","Orange", "lightblue","mistyrose"], AlPoint=(1.8444444444444446,5.311111111111111),Utility_Show = True, Utility_draw = True, marginal_rate=1,al_color = "red")
    draw_Ed_Bow(U1,U2,6,6,colors=["k","Orange", "lightblue","mistyrose"], AlPoint=(1.6989247311827957,5.311827956989247),Utility_Show = True, Utility_draw = False, marginal_rate=0.9,al_color = "orange")
    draw_Ed_Bow(U1,U2,6,6,colors=["k","Orange", "lightblue","mistyrose"], AlPoint=(1.826086956521739,5.532608695652174),Utility_Show = True, Utility_draw = False, marginal_rate=0.8,al_color = "gold")
    draw_Ed_Bow(U1,U2,6,6,colors=["k","Orange", "lightblue","mistyrose"], AlPoint=(2.098901098901099,5.428571428571429),Utility_Show = True, Utility_draw = False, marginal_rate=0.7,al_color = "lawngreen")
    draw_Ed_Bow(U1,U2,6,6,colors=["k","Orange", "lightblue","mistyrose"], AlPoint=(1.8275862068965518,3.7758620689655173),Utility_Show = True, Utility_draw = False, marginal_rate=0.6,al_color = "green")
    draw_Ed_Bow(U1,U2,6,6,colors=["k","Orange", "lightblue","mistyrose"], AlPoint=(2.2096774193548385,4.532258064516129),Utility_Show = True, Utility_draw = False, marginal_rate=0.5,al_color = "deepskyblue")
    draw_Ed_Bow(U1,U2,6,6,colors=["k","Orange", "deepskyblue","salmon"], AlPoint=(2.7222222222222223,3.988888888888889),Utility_Show = True, Utility_draw = True, marginal_rate=0.4,al_color = "cornflowerblue")
    draw_Ed_Bow(U1,U2,6,6,colors=["k","Orange", "lightblue","mistyrose"], AlPoint=(1.9870129870129871,4.534090909090909),Utility_Show = True, Utility_draw = False, marginal_rate=0.3,al_color = "navy")
    draw_Ed_Bow(U1,U2,6,6,colors=["k","Orange", "lightblue","mistyrose"], AlPoint=(2.3295454545454546,5.311111111111111),Utility_Show = True, Utility_draw = False, marginal_rate=0.2,al_color = "slateblue")
    draw_Ed_Bow(U1,U2,6,6,colors=["k","Orange", "lightblue","mistyrose"],  Xlab=r"$X^1$",Ylab=r"$X^2$", AlPoint=(2.1666666666666665,4.433333333333334),Utility_Show = True, Utility_draw = False, marginal_rate=0.1,al_color = "violet")

    plt.legend()
    plt.savefig("./indiffernce_curve_experiment.png")