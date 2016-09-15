# Portafolio
Proyecto de portafolio 6 activos Markowitz

# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

import numpy as np;
import matplotlib.pyplot as plt
import pandas.io.data as wb;
import datetime;

#%%
#Descargas precios de yahoo
start = datetime.datetime(2016,1,1);
end = datetime.datetime(2016,9,14);
grum = wb.DataReader("GRUMAB.MX",'yahoo',start,end);
bimb = wb.DataReader("BIMBOA.MX",'yahoo',start,end);
sori = wb.DataReader("SORIANAB.MX",'yahoo',start,end);

#%%
Prgrum = grum.values[:,5];
Prbimb = bimb.values[:,5];
Prsori = sori.values[:,5];

ndata=Prgrum.size;

#Calcular rendimientos
rendgrum = (Prgrum[1:ndata]/Prgrum[0:ndata-1])-1;
rendbimb = (Prbimb[1:ndata]/Prbimb[0:ndata-1])-1;
rendsori = (Prsori[1:ndata]/Prsori[0:ndata-1])-1;

data = np.zeros((ndata-1,3))
data[:,0]=rendgrum;
data[:,1]=rendbimb;
data[:,2]=rendsori;

rende = np.mean(data,0);
covs = np.cov(data,rowvar=False);

#%%
npart = 10;
ntemp,nact = np.shape(data);
"""
# Crear datos para dibujar la función de rastrigin
"""
def Markow(part,rende,covs):
    npart, nact = np.shape(part)    ;
    rendm = np.matrix(rende);
    covm = np.matrix(covs);
    partm = np.matrix(part);
    
    #Calcular los rendimientos
    rendp = partm*np.transpose(rendm);
    riskp = partm*covm*np.transpose(partm);
    riskp = np.transpose(riskp[np.arange(0,npart),np.arange(0,npart)]);
    return np.array(np.transpose(rendp)),np.array(np.transpose(riskp))

def porta(x1,x2,x3,rende,covs):
    nx = np.size(x1);
    restric1 = np.zeros(nx);
    restric2 = np.zeros(nx);
    restric3 = np.zeros(nx);
    restric4 = np.zeros(nx);
    restric5 = np.zeros(nx);
    restric6 = np.zeros(nx);
    restric7 = np.zeros(nx);
    alfa1 = 1000;
    alfa2 = 1000;
    alfa3 = 1000;
    alfa4 = 1000;
    alfa5 = 1000;
    alfa6 = 1000;
    alfa7 = 1000;
    #x1>=0
    index = ((-x1)>0);
    restric1[index] = (-x1[index])*alfa1;
    #x1-1<=0
    index = ((x1-1)>0);
    restric2[index] = (x1[index]-1)*alfa2;
    #x2>=0
    index = ((-x2)>0);
    restric3[index] = (-x2[index])*alfa3;
    #x2-1<=0
    index = ((x2-1)>0);
    restric4[index] = (x2[index]-1)*alfa4;
    #x1+2*x2<=10
    restric5 = np.abs(x1+x2+x3-1)*alfa5; 
    #x3>=0
    index = ((-x3)>0);
    restric6[index] = (-x3[index])*alfa6;
    #x3-1<=0
    index = ((x3-1)>0);
    restric7[index] = (x3[index]-1)*alfa7;
    
    beta1=0;
    beta2=1;
    
    part = np.zeros((nx,3));
    part[:,0] = x1;
    part[:,1] = x2;
    part[:,2] = x3; 
    Rp,Sp = Markow(part,rende,covs)
    return -beta1*Rp+beta2*Sp+restric1+restric2+restric3+restric4+restric5+restric6+restric7;

#%%
"""
# Crear las particulas iniciales
"""

# Inicializar variables para el enjambre de particulas
npart = 100;
niter = 10000;
c1 = 0.001;
c2 = 0.001;

x1 = np.random.rand(npart);
x1pl = x1;
x1pg = 0;
vx1 = np.zeros(npart);

x2 = np.random.rand(npart);
x2pl = x2;
x2pg = 0;
vx2 = np.zeros(npart);

x3 = np.random.rand(npart);
x3pl = x3;
x3pg = 0;
vx3 = np.zeros(npart);

#%%
fpl = 1000000*np.ones(npart);
fpg = 1000000;

for k in range(0,niter):
    
    # evaluación de la función a minimizar
    fp = porta(x1,x2,x3,rende,covs);
    fp = np.reshape(fp,npart);
    # Encontrar el minimo global
    index = np.argmin(fp)
    if (fp[index] < fpg):
        x1pg = x1[index];
        x2pg = x2[index];
        x3pg = x3[index];
        fpg = fp[index];
        
    # Encontrar el minimo local de cada particulo
    for ind in range(0,npart):
        if (fp[ind] < fpl[ind]):
            x1pl[ind] = x1[ind];
            x2pl[ind] = x2[ind];
            x3pl[ind] = x3[ind];
            fpl[ind] = fp[ind];
            
    
    # Mover las particulas según las ecuaciones de movimiento
    vx1=vx1 + c1*np.random.rand(npart)*(x1pg-x1)\
    + c2*np.random.rand(npart)*(x1pl-x1)
    x1 = x1 + vx1;
    vx2=vx2 + c1*np.random.rand(npart)*(x2pg-x2)\
    + c2*np.random.rand(npart)*(x2pl-x2)
    x2 = x2 + vx2;
    vx3=vx3 + c1*np.random.rand(npart)*(x3pg-x3)\
    + c2*np.random.rand(npart)*(x3pl-x3)
    x3 = x3 + vx3;

#%%
#X1 = np.arange(-10,10,0.01);
#X2 = np.arange(-10,10,0.01);
#X1,X2 = np.meshgrid(X1,X2);
#X1V = np.reshape(X1,4000000);
#X2V = np.reshape(X2,4000000);
#Z = porta(X1V,X2V);
#Z = np.reshape(Z,(2000,2000));
#plt.contour(X1,X2,Z,20);

#%%
plt.figure(1);
plt.plot(x1,x2,'b.',x1pg,x2pg,'go',0,0,'ro');
plt.plot([0,0],[-10,10],'k--',[-10,10],[0,0],'k--',[0,1],[1,0],'k--',[1,1],[-10,10],'k--',[-10,10],[1,1],'k--');
plt.xlabel('X1');
plt.ylabel('X2');
plt.axis([-0.5,1.5,-0.5,1.5]);
plt.title('Resultados: X1 = %.4f, X2 = %.4f, X3 = %.4f, fp = %.4f ' % (x1pg,x2pg,x3pg,fpg));
plt.show();
print('Resultados: X1 = %.4f, X2 = %.4f, X3 = %.4f, fp = %.4f '% (x1pg,x2pg,x3pg,fpg));
