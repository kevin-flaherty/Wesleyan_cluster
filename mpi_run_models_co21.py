#~/usr/bin/env python

#Identical to run_models.py, except using open mpi functionality to parallelize code, rather than thread keyword

# run this code on a home machine with mpirun -np 2 python mpi_run_models.py


from single_model import *
import numpy as np
import emcee
import sys
from emcee.utils import MPIPool


dir = str(sys.argv[1])+'/'

# Set the number of walkers and dimensions
#   number of walkers must be even
nwalkers = 80#200

#CO(2-1)
names=['qq','Rc','vturb','Tatm','incl','Rin','vsys']#,'qvturb']
ndim = len(names)
p0 = np.zeros((nwalkers,ndim))
p0[:,0] = np.random.rand(nwalkers)*(-.1+1.)-1.      #qq 
p0[:,1] = np.random.rand(nwalkers)*(2.5-1.9)+1.9     #log(Rc)
p0[:,2] = np.random.rand(nwalkers)*(1.-.0001)+.0001 #vturb
p0[:,3] = np.random.rand(nwalkers)*(80-17)+17       #Tatm
p0[:,4] = np.random.rand(nwalkers)*(-40+25.)-25      #incl
p0[:,5] = np.random.rand(nwalkers)*(20.-1)+1.       #Rin
p0[:,6] = np.random.rand(nwalkers)*(6.2-5.8)+5.8    #vsys


# Initialize the pool
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit()

# Initialize the sampler with the chosen specs
#highres=False, massprior=False, cleanup=True, systematic=True,line='co21',vcs=True,exp_temp=False,add_ring=False,save_baselines=False
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnlike,args=[True,False,True,False,'co21',True,False,False,False],pool=pool,a=1.5) 


 
# Run Nburn steps as a burn-in
#nburn = 1#120
#pos, prob, state = sampler.run_mcmc(p0,nburn)
#sampler.reset()

# Run N steps, with breaks to save the data
Nsteps = 800#300
pos,prob,state=sampler.run_mcmc(p0,Nsteps)

for iparam in range(ndim): 
    file = dir+'chain_dmtau_co21_han_'+names[iparam]+'.dat'
    f = open(file,'w')
    data = (sampler.chain[:,:,iparam]).squeeze() #dimensions of the chain are Nwalkers*Nsteps*Ndim
    for j in range(Nsteps):
        line = str(j)+' '+(' '.join(format(x,'5.6f') for x in data[:,j]))
        f.write(line+'\n')
 
    f.close()

file = '/home/kflaherty/dmtau/lnlike_dmtau_co21_han.dat'
f = open(file,'w')
line=' '.join(format(x,'5.6f') for x in prob)
f.write(line+'\n')
f.close()

# format of table is Step#, Position of Walker 1, Position of Walker 2, etc.
                         

#close the processes
pool.close()



