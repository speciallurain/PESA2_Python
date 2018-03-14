#--coding:utf-8--
#!/usr/bin/python
# -*- coding: utf-8 -*-
#PESA2算法 by Lo Rain ,qq:771527850,E-mail:luyueliang423@163.com
import math
import numpy as np
import random
import numpy
import matplotlib.pyplot as plt
import matplotlib
Num_object,nVar,VarMin,VarMax,MaxIt,nPop,nArchive,nGrid,InflationFactor,beta_deletion,beta_selection,pCrossover=3,30,0.0001,0.9999,50,30,200,10,0.1,1,2,0.7

###########################################################
class crossover_param(object):
    def __init__(self,VarMin,VarMax):
       self.gamma=0.1
       self.VarMin=VarMin
       self.VarMax=VarMax
crossover_params=crossover_param(VarMin,VarMax)
############################################################
############################################################
class mutation_param(object):
    def __init__(self, VarMin, VarMax):
        self.h = 0.3
        self.VarMin = VarMin
        self.VarMax = VarMax
mutation_params=mutation_param(VarMin,VarMax)
###########################################################
###########################################################
def random_int_list(start, stop, length):    # return the matrix of random
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.uniform(start, stop))
        random_list.sort()
    return random_list
###########################################################
###########################################################
def CostFunction(x):    #fitness function for two objects
    n = 0
    for i in range(len(x)):
        if i > 1:
            n = (x[i] - 0.5) ** 2 - math.cos(20 * math.pi * (x[i] - 0.5))
    g = 100 * (len(x) - 3 + n)
    # f1=(0.5*t*(1+g))
    # f2=(0.5*t2*(1+g)*(1-x[len(x)-2]))
    # f3= (0.5*t3*(1+g)*(1-x[len(x)-3]))

    f1 = 0.5 * x[0] * x[1] * (1 + g)
    f2 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
    f3 = 0.5 * (1 - x[0]) * (1 + g)
    # print f2,f3
    return [f1, f2, f3]
    # z1=1-math.exp(-(((x[0]-1/math.sqrt(len(x)))**2+(x[1]-1/math.sqrt(len(x)))**2+(x[2]-1/math.sqrt(len(x)))**2)))
    # z2=1-math.exp(-(((x[0]+1/math.sqrt(len(x)))**2+(x[1]+1/math.sqrt(len(x)))**2+(x[2]+1/math.sqrt(len(x)))**2)))
    # #z1=x[0]
    # #z2=1+9/(len(x)-1)*(x[1]+x[2])
    # return [z1,z2]
###########################################################
###########################################################
def Dominates(x,y):
    # b=all([x[0]<=y[0],x[1]<=y[1]])and any([x[0]<y[0],x[1]<y[1]])
    b = 0
    if float(x[0]) <= float(y[0]):
        if float(x[1]) <= float(y[1]):
            if float(x[2]) <= float(y[2]):
                if float(x[0]) < float(y[0]):
                    b = 1
                elif float(x[1]) < float(y[1]):
                    b = 1
                elif float(x[2]) < float(y[2]):
                    b = 1
    return b
###########################################################
###########################################################
def DetermineDomination(pop_Cost):    #for determine the dominatied pop in the orginal pop
    n=len(pop_Cost)
    pop_IsDominated1=[]
    for i in range(n):
        pop_IsDominated1.append(0)
    for i in range(n):
        #if pop_IsDominated[i]:
        #    continue
        for j in range(n):
            if j!=i:
                if Dominates(pop_Cost[j],pop_Cost[i]):
                    pop_IsDominated1[i]=1    #if the pop dominatied we write the tital "1"
                    break
    #print len(pop_IsDominated1)
    #print len(pop_Cost)
    #for i in range(len(pop_IsDominated1)):
       # if pop_IsDominated1[i]==1:
           # for j in range(len(pop_Cost)):
                #if j!=i:
                  #  if (pop_Cost[j][0]==pop_Cost[i][0]):
                     #   if (pop_Cost[j][1]==pop_Cost[i][1]):
                       #     pop_IsDominated1[i]=0

    return pop_IsDominated1
   # print pop_IsDominated
    #return pop_IsDominated
############################################################
############################################################
def Creategrid(pop_Cost,nGrid,InflationFactor):
    zmin = [min([t[0] for t in pop_Cost]), min([t[1] for t in pop_Cost]),
            min([t[2] for t in pop_Cost])]  # constract the min and max value of each objects
    zmax = [max([t[0] for t in pop_Cost]), max([t[1] for t in pop_Cost]), max([t[2] for t in pop_Cost])]
    dz = [zmax[0] - zmin[0], zmax[1] - zmin[1],
          zmax[2] - zmin[2]]  # constract the distrance of min and max value of each objects
    alpha = InflationFactor / 2.0
    zmin = [zmin[0] - alpha * dz[0], zmin[1] - alpha * dz[1], zmin[2] - alpha * dz[2]]
    zmax = [zmax[0] + alpha * dz[0], zmax[1] + alpha * dz[1], zmax[2] + alpha * dz[2]]

    nObj = len(zmin)
    C1, C2, C3 = [], [], []
    x = numpy.linspace(zmin[0], zmax[0], nGrid + 1)  # obtain the Grid of two objects
    for k in range(len(x)):
        C1.append(x[k])
    x = numpy.linspace(zmin[1], zmax[1], nGrid + 1)
    for k in range(len(x)):
        C2.append(x[k])
    x = numpy.linspace(zmin[2], zmax[2], nGrid + 1)
    for k in range(len(x)):
        C3.append(x[k])
    C = [C1, C2, C3]
    empty_grid_N = numpy.zeros([len(C1), len(C2), len(C3)])
    t = 0
    for i in range(len(C1)):
        for j in range(len(C2)):
            for k in range(len(C3)):
                empty_grid_N[i][j][k] = t
                t = t + 1
    pop_Grid = numpy.zeros(len(pop_Cost))
    grid = numpy.zeros(t)
    for i in range(len(pop_Cost)):
        t1 = numpy.zeros(len(C1))
        t2 = numpy.zeros(len(C2))
        t3 = numpy.zeros(len(C2))
        for k in range(len(C1)):
            if pop_Cost[i][0] <= C1[k]:
                t1[k] = 1
        for k in range(len(C1)):
            if pop_Cost[i][1] <= C2[k]:
                t2[k] = 1
        for k in range(len(C1)):
            if pop_Cost[i][2] <= C3[k]:
                t3[k] = 1
        # t1 = pop_Cost[i][0] <= numpy.array[C1]  # we must add the numpy.array,if not,it will putout only one value
        # t2 = pop_Cost[i][1] <= numpy.array[C2]
        # t3=  pop_Cost[i][2] <= numpy.array[C3]
        m = 0
        n = 0
        o = 0
        for k in range(len(C1)):
            if t1[k]:
                break
            else:
                m = m + 1
        for j in range(len(C2)):
            if t2[j]:
                break
            else:
                n = n + 1
        for j in range(len(C3)):
            if t3[j]:
                break
            else:
                o = o + 1
        pop_Grid[i] = empty_grid_N[m - 1][n - 1][o - 1]
        grid[int(empty_grid_N[m - 1][n - 1][o - 1])] = grid[int(
            empty_grid_N[m - 1][n - 1][o - 1])] + 1  # because empty_grid_N[m-1][n-1] is flourt, so we must add the int
    return [pop_Grid, grid]
#############################################################
#############################################################
def TruncatePopulation(archive_Position,archive_Cost,archive_Grid,nArchive):

    while len(archive_Grid)-nArchive>0:   #in the fellow,we will select the pop in archive,for contralling the number of pop in archive
        t_i=np.argmax(grid)    #the positon of the pop
        t_value=max(grid)      #the number of pop in archive
        select_value=int(t_value*random.random())   #creat a random number for delete the pop
        t=0
        ar_Grid = []
        ar_Position = []
        ar_Cost = []
        for j in range(len(archive_Cost)):
            if archive_Grid[j]==t_i:
                if t!=select_value:                 ##in the archive which have the most pop,we will delect one pop randomly
                    ar_Cost.append(archive_Cost[j])
                    ar_Grid.append(archive_Grid[j])
                    ar_Position.append(archive_Position[j])
                    t=t+1
                else:
                    grid[t_i]=grid[t_i]-1
                    t=t+1
            else:
                ar_Cost.append(archive_Cost[j])
                ar_Grid.append(archive_Grid[j])
                ar_Position.append(archive_Position[j])
        archive_Position=ar_Position    #obtain the new archive
        archive_Grid=ar_Grid
        archive_Cost=ar_Cost
    return archive_Cost,archive_Grid,archive_Position
####################################################################
####################################################################
def RouletteWheelSelection(p):
    r=random.random()*sum(p)
    c=numpy.cumsum(p)
    t=r>c
    i=0
    for j in range(len(t)):
        if t[j]:
            i=i+1
        else:
            break
    return i
####################################################################
####################################################################
def SelectFromPopulation(archive,grid,beta): #this function,maybe you look that is so complex ,but it just obtain a random value
    sg=grid[grid>0]
    p1=numpy.zeros(len(sg))
    p = numpy.zeros(len(sg))
    for i in range(len(sg)):
        p1[i]=(1/float(sg[i]))**beta
    for i in range(len(sg)):
        p[i]=p1[i]/float(sum(p1))
    k=RouletteWheelSelection(p)
    t=0
    m=0
    mm=0
    kk=0
    for i in range(len(grid)):
        if grid[i]!=0:
            if t==k:
                for j in range(len(archive)):
                   if archive[j]==i:
                       m=m+1
                for j in range(len(archive)):
                    tt=int((m-1)*random.random())
                    if archive[j]==i:
                        if tt==mm:
                            kk=j
                            break
                        else:
                            mm = mm + 1
                break
            else:
                t=t+1
    return kk
#####################################################################
#####################################################################
def Crossover(x1,x2,params):
    gamma=params.gamma
    VarMin=params.VarMin
    VarMax=params.VarMax
    alpha=random_int_list(-gamma,1+gamma,len(x1))
    y1=[]#numpy.multiply(alpha,x1)+numpy.multiply((1-alpha),x2)
    y2=[]#numpy.multiply(alpha,x2)+numpy.multiply((1-alpha),x1)
    for i in range(len(alpha)):
        y1.append(alpha[i]*x1[i]+(1-alpha[i])*x2[i])
        y2.append(alpha[i]*x2[i]+(1-alpha[i])*x1[i])
        y1[i]=max(y1[i],VarMin)
        y1[i] =min(y1[i], VarMax)
        y2[i] =max(y2[i], VarMin)
        y2[i] =min(y2[i], VarMax)
    return [y1,y2]
#####################################################################
#####################################################################
def Mutate(x,params):
    h=params.h
    VarMin=params.VarMin
    VarMax=params.VarMax
    sigma=h*(VarMax-VarMin)
    a = numpy.random.randn(len(x))
    y = x + sigma * a

    for i in range(len(y)):
        y[i]=max(y[i],VarMin)
        y[i]=min(y[i],VarMax)
    return y
#####################################################################
#####################################################################
total_archive_Cost=[]
total_archive_Position=[]

for test2 in range(30):
    print test2
    VarSize=[nVar,1]    #Decision Variables Matrix Size
    Vdis=(VarMax-VarMin)/float(nVar)
    nObj=Num_object    #(CostFunction(random.randrange(VarMin,Varmax,Vdis))).size
    ######PESA-II Settings
    nCrossover=round(pCrossover*nPop/2.0)*2
    pMutation=1-pCrossover
    nMutation=nPop-nCrossover
    pop_Position=[]
    pop_Cost=[]
    for i in range(nPop):    #initialise the pop
        pop_Position.append(random_int_list(VarMin,VarMax,nVar))
    for i in range(nPop):
        pop_Cost.append(CostFunction(pop_Position[i]))

    archive_Position=[]
    archive_Cost=[]
    ndpop=[]
    ndpop_Position=[]
    ndpop_Cost=[]
    #################################Main LOOP################################################
    for it in range(MaxIt):
        #print 'Iterate',it
        ndpop_Position = []    #clear the transfer matrix
        ndpop_Cost = []
        #pop_IsDominated =[]
        #select the dominated pop from the origal pop


        pop_IsDominated=DetermineDomination(pop_Cost)

        for i in range(len(pop_Cost)):
            if pop_IsDominated[i]==0:
                ndpop_Position.append(pop_Position[i])
                ndpop_Cost.append(pop_Cost[i])
        for k in range(len(ndpop_Cost)):
           # mt=0
          #  for j in range(len(archive_Cost)):
             #   if archive_Cost[j][0]==ndpop_Cost[k][0]:
               #     if archive_Cost[j][1]==ndpop_Cost[k][1]:
                   #     mt=1
                  #      break

            #if mt==0:
            archive_Position.append(ndpop_Position[k])
            archive_Cost.append(ndpop_Cost[k])


        #delete the dominated pop in the archive
        ndpop_Position = []
        ndpop_Cost = []
        pop_IsDominated=DetermineDomination(archive_Cost)
        for i in range(len(archive_Cost)):
            if pop_IsDominated[i]==0:
                ndpop_Position.append(archive_Position[i])
                ndpop_Cost.append(archive_Cost[i])
        #print ndpop_Cost

        archive_Cost=[]
        archive_Position=[]
        for k in range(len(ndpop_Cost)):
            mt=0
            for j in range(len(archive_Cost)):
                #print len(archive_Cost)
                if k!=1:
                    if archive_Cost[j][0]==ndpop_Cost[k][0]:
                        if archive_Cost[j][1]==ndpop_Cost[k][1]:
                            if archive_Cost[j][2]==ndpop_Cost[k][2]:
                                mt=1
                                break

            if mt==0:
                archive_Position.append(ndpop_Position[k])
                archive_Cost.append(ndpop_Cost[k])

        #Now,in archive_cost,no pop dominated by each other
        [archive_Grid,grid]=Creategrid(archive_Cost,nGrid,InflationFactor)
        if len(archive_Cost)>nArchive:
            archive_Cost,archive_Grid,archive_Position=TruncatePopulation(archive_Position,archive_Cost,archive_Grid,nArchive)


        #[archive_Grid, grid] = Creategrid(archive_Cost, nGrid, InflationFactor)
        #  #if we have deleted the pop which have the max value,the grid in the environment will be changed

        #in the follow, we will in Crossover behavior
        pop_Position=[]
        pop_Cost=[]
        #print len(archive_Cost)
        for c in range(int(nCrossover/2.0)):
            #print c

            p1=SelectFromPopulation(archive_Grid,grid,beta_selection)
            p2=SelectFromPopulation(archive_Grid,grid,beta_selection)

            [popc_1,popc_2]=Crossover(archive_Position[p1],archive_Position[p2],crossover_params)
            pop_Position.append(popc_1)
            pop_Position.append(popc_2)
            pop_Cost.append(CostFunction(popc_1))
            pop_Cost.append(CostFunction(popc_2))
        # in the follow,we wil continue the Mutation behavior
        for m in range(int(nMutation)):
            p= SelectFromPopulation(archive_Grid, grid, beta_selection)
            popm_1=Mutate(archive_Position[p],mutation_params)
            pop_Position.append(numpy.array(popm_1))
            pop_Cost.append(CostFunction(popm_1))
    # x = [t[0] for t in archive_Cost]
    # y = [t[1] for t in archive_Cost]
    #     # print x
    #     # print y
    #
    # plt.figure(11)
    # plt.plot(x, y, 'ro')
    # plt.xlabel('f_1')
    # plt.ylabel('f_2')
    # plt.show()
    # from mpl_toolkits.mplot3d import Axes3D
    # x = [t[0] for t in archive_Cost]
    # y = [t[1] for t in archive_Cost]
    # z = [t[2] for t in archive_Cost]
    # fig = plt.figure(11)
    # ax = Axes3D(fig)
    # ax.scatter(x, y, z, c='r')  # 绘点
    # # plt.plot(x, y, 'ro')
    # plt.xlabel('f_1')
    # plt.ylabel('f_2')
    # plt.show()
    total_archive_Cost.append(archive_Cost)
    total_archive_Position.append(archive_Position)
# this is the test part,for testing which is the wrong part
   # print nMutation
   # for i in range(len(pop_Cost)):
      #  for k in range(len(pop_Cost)):

       #    if pop_Cost[i][0]==pop_Cost[k][0]:
         #       if pop_Cost[i][1]==pop_Cost[k][1]:
        #            if k!=i:
         #               pop_Cost[i]=[0,0]
   # print pop_Cost
###################################################

mydata = total_archive_Cost

thefile= open("DTLZ1_PESA2_COST.txt", "w+")
for item in mydata:
  thefile.write("%s\n"% item)
thefile.close()
mydata=[]
mydata1 = total_archive_Position
thefile= open("DTLZ1_PESA2_Position.txt", "w+")
for item in mydata1:
  thefile.write("%s\n"% item)
thefile.close()
# for i in range(3):
#     myfile.write(total_archive_Cost[i] + '\n')
# x=[t[0] for t in archive_Cost]
# y=[t[1] for t in archive_Cost]
# #print x
# #print y
#
# plt.figure(11)
# plt.plot(x,y, 'ro')
# plt.xlabel('f_1')
# plt.ylabel('f_2')
# plt.show()











# the fellow is test part,must delete in the end


