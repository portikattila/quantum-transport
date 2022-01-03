#import libraries
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.linalg as sl

from IPython.display import display
from ipywidgets import*

class dirty_edge_QWZ_model:
    '''
    QWZ model class
    '''
    def __init__(self, L, delta):
        '''
        Initializing the class
        '''
        
        self.L = L 
        self.d = delta
        
        self.tol = 1e-8
        
        #pauli matrices
        self.sigma_0 = np.array([[1,0],[0,1]])
        self.sigma_x = np.array([[0,1],[1,0]])
        self.sigma_y = np.array([[0,-1j],[1j,0]])
        self.sigma_z = np.array([[1,0],[0,-1]])
        
        
        self.U = delta * self.sigma_z
        self.T_x = 0.5 * (self.sigma_z + 1j*self.sigma_x)
        self.T_y = 0.5 * (self.sigma_z + 1j*self.sigma_y)
    

################ QWZ lead #############################
        
    def create_lead(self):
        '''QWZ lead''' 
        
        self.U = self.d * self.sigma_z
        Hu = np.kron(np.eye(self.L), self.U)
        
        #one unit cell
        h0 = np.eye(self.L, k=1)
        H0 = np.matrix(np.kron(h0, self.T_x))
        self.H0 = Hu + H0 + H0.H
        self.dim = np.shape(H0)[0]
        
        #relations between unit cells
        h1 = np.eye(self.L)
        #---------singular matrix...----------------- small diagonal matrix to solve singular H1 matrix problem
        self.H1 = np.matrix(np.kron(h1, self.T_y)) + 1e-2*np.eye(self.dim) 
        
    
    def lead_spectrum(self, K):
        '''calculate the lead's spectrum as a function of mumentum'''
        #create lead
        self.create_lead()
        
        l = len(K)
        S = np.zeros((l,self.dim))
        for i in range(l):
            k = K[i]
            S[i,:]=np.linalg.eigvalsh(self.H0 + np.exp(1.0j*k) * self.H1+np.exp(-1.0j*k)*self.H1.H)
            
        return S 
    
            
    def leads_Green(self,E,**kwargs):
        '''
        Calculate the lead's Green's functions
            based on mezo.py
        '''
             
        O = np.zeros_like(self.H0) #zero block                
        I = np.eye(self.dim) # identity block 
        H1_inv = np.linalg.inv(self.H1)
        
        #eigenvalue problem

        M1 = np.vstack((np.hstack((E*np.eye(self.dim)-self.H0,-self.H1.H)),
                       np.hstack((I,O))))        
        M2 = np.vstack((np.hstack((self.H1,O)),
                       np.hstack((O,I))))
                  
        #calculate the eigen values and vectors
        w,v = sl.eig(M1,M2)
        self.w = w
        v = np.matrix(v[:self.dim,:]) #ket phi (upper part)
        
        gv=np.zeros_like(w, dtype=complex)
        for i in range(2*self.dim):
            v[:,i] = v[:,i]/np.linalg.norm(v[:,i])  #normalized eigenstate
            gv[i] = 1.0j*((v[:,i].H*(w[i]*self.H1-w[i]**(-1)*self.H1.H)*v[:,i])[0,0]) #group velocity
            
        self.gv = gv
        
        index = np.arange(len(w))
        #-------------------------------------------opened--------------------------------------------------closed
        left = index[np.logical_or(np.logical_and(abs(abs(w)-1)<self.tol, np.real(gv)<0), np.logical_and(abs(abs(w)-1)>self.tol, abs(w)>1))]
        right = index[np.logical_or(np.logical_and(abs(abs(w)-1)<self.tol, np.real(gv)>0), np.logical_and(abs(abs(w)-1)>self.tol, abs(w)<1))]

        
        self.v_left = v[:,left]
        self.w_left  = w[left]
        self.gv_left = gv[left]
        
        self.v_right = v[:,right]
        self.w_right = w[right]
        self.gv_right =gv[right]
        
        if len(self.w_right)!=len(self.w_left):
            print('Problem with partitioning!!', len(self.w_right), len(self.w_left))
            return
        
        #calculate the dual vectors and calculate the preliminaries to the green function
        self.v_right_dual = np.linalg.inv(self.v_right)
        self.v_left_dual  = np.linalg.inv(self.v_left)          
        
        self.T_left    = self.v_left  @ np.matrix(np.diag(1/self.w_left))  @ self.v_left_dual
        self.T_right   = self.v_right @ np.matrix(np.diag(self.w_right))   @ self.v_right_dual

        #Get Green's functions leads
        self.SG_L = self.T_left  * np.linalg.inv(self.H1)
        self.SG_R = self.T_right * np.linalg.inv(self.H1.H)

        
        
################ Bulk sample #############################

    def create_hamiltonian(self, W, alpha=0, V_size=0, V_pos=None, **kwargs):
        '''Create the hamiltonian of a finite rectangle sample and add a scatter (as a random matrix)'''
        
        if W:
            self.W = W # width of the sample
        else:
            self.W = self.L
            
        self.s = self.L * self.W # the size
        
        #size and position of V
        try:
            self.VW = V_size[0]
            self.VL = V_size[1]
        except:
            self.VW = self.VL = V_size
            
        self.Vs = self.VW * self.VL
    
        if V_pos:
            try:
                self.Vx = V_pos[0]
                self.Vy = V_pos[1]
            except:
                self.Vx = self.Vy = pos
        else:
            self.Vx = 0
            self.Vy = np.floor((self.W-self.VW)/2)
            
        x_range = np.arange(self.L) 
        y_range = np.arange(self.W)
        y_coordinates, x_coordinates = np.meshgrid(y_range, x_range)
        (self.y, self.x) = (y_coordinates.flatten(),x_coordinates.flatten()) #coordinates to mark the place of the scattering part

        #hamiltonian
        self.H = np.kron(np.eye(self.W), self.H0) + np.kron(np.eye(self.W,k=1), self.H1) + np.kron(np.eye(self.W,k=-1), self.H1.H)

        #add the scatterer
        if self.Vs > 0:
            self.V_x_coord = self.Vx + np.array([0,self.VL-1,self.VL-1,0,0]) #the coordinates of the scatterer
            self.V_y_coord = self.Vy + np.array([0,0,self.VW-1,self.VW-1,0])
        
            V = matplotlib.path.Path(np.array([self.V_x_coord, self.V_y_coord]).T)        
            
            #scaterer
            self.lattice_index=(V.contains_points(
                               np.array([self.x,self.y]).T, radius=0.1))
            
            self.matrix_index=(V.contains_points(
                               np.array([np.kron(self.x,np.ones(2)),
                                         np.kron(self.y,np.ones(2))]).T, radius=0.1))
           
            #outside
            self.outside_lattice_index = np.logical_not(self.lattice_index)
            self.outside_matrix_index = np.logical_not(self.matrix_index)

            self.V = self.H[:,self.matrix_index][self.matrix_index,:]
            self.mask = np.kron(self.matrix_index.reshape((self.matrix_index.size,1)), self.matrix_index) #the covered part of the original matrix
            
            if kwargs.get('cut', False):
                self.H[self.mask]=0
                self.H_S = self.H
            
            elif kwargs.get('random', False):
                R0 = np.random.randn(*self.V.shape)+np.random.rand(*self.V.shape)*1.0j 
                self.R0 = alpha * (R0+np.transpose(np.conjugate(R0))).flatten() # random unitary matrix
                self.R = np.zeros_like(self.H)
                self.R[self.mask] = self.R0 #embedding to a zero matrix
                self.H_S = self.H + self.R
            
            else:
                self.H[self.mask]=0
                R0 = np.random.randn(*self.V.shape)+np.random.rand(*self.V.shape)*1.0j 
                self.R0 = alpha * (R0+np.transpose(np.conjugate(R0))).flatten() # random unitary matrix
                self.R = np.zeros_like(self.H)
                self.R[self.mask] = self.R0 #embedding to a zero matrix
                self.H_S = self.H + self.R
                
        else:    
            self.H_S = self.H
            
        

    def lattice_plot(self):
        '''function to plot the scatterer on lattice'''
        fig, ax = plt.subplots(1,1,figsize=(8,8))

        if self.Vs > 0:
            ax.plot(self.V_x_coord,self.V_y_coord, 'r-')
            ax.fill(self.V_x_coord,self.V_y_coord, color='r', alpha=0.6)
            ax.text(self.Vx-0.5+self.VL/2, self.Vy-0.5+self.VW/2, '$\hat{V}$', ha='center', va='center', fontsize=30)
            ax.plot(self.x[np.logical_not(self.lattice_index)], self.y[np.logical_not(self.lattice_index)], 'ko')
            ax.plot(self.x[self.lattice_index], self.y[self.lattice_index], 'o', color='tab:red')
        else:
            ax.plot(self.x, self.y, 'ko')
            
        ax.set_xticks(np.arange(0,self.L,1))
        ax.set_xticklabels(np.arange(0,self.L,1) + 1)
        ax.set_yticks(np.arange(0,self.W,1))
        ax.set_yticklabels(np.arange(0,self.L,1) + 1)
        ax.tick_params(size=10, direction='inout', width=2, labelsize=15)

        plt.show()
        

    def spectrum_and_density(self):
        ''''''
        e,v = np.linalg.eigh(self.H_S)
    
        return e,v
    
    def Green(self, E):
        ''' '''
        
        self.E = E
        self.leads_Green(self.E) 
        self.SG_B = np.linalg.inv(self.E * np.eye(2*self.s)-self.H_S) #the Green's function of the scaterer
        
        #decoupled Green's functions
        self.G0 = sp.linalg.block_diag(self.SG_L, self.SG_B, self.SG_R)
        

    def Dyson(self):
        '''Dyson equation'''
        
        # coupling
        self.V = np.vstack([np.hstack([np.zeros_like(self.H1) ,self.H1, np.zeros((self.H1.shape[0], self.G0.shape[1]-2*self.H1.shape[1]))]),
                            np.hstack([self.H1.H,  np.zeros((self.H1.shape[0], self.G0.shape[1]-self.H1.shape[1]))]),
                            np.zeros((self.G0.shape[1]-4*self.H1.shape[1], self.G0.shape[1])),
                            np.hstack([np.zeros((self.H1.shape[0], self.G0.shape[1]-self.H1.shape[1])), self.H1]),
                            np.hstack([np.zeros((self.H1.shape[0], self.G0.shape[1]-2*self.H1.shape[1])),self.H1.H,np.zeros_like(self.H1)])])
        
        self.G = np.linalg.inv(np.linalg.inv(self.G0) - self.V)

        
    def LDOS(self, E, delta=-1, alpha=1, W=None, V_size=(6,3), V_pos=None, **kwargs):
        '''Local density of states as a function of energy'''
        
        
        self.U = delta * self.sigma_z
        self.create_lead()
        self.create_hamiltonian(W,alpha, V_size=V_size, V_pos=V_pos, random=kwargs.get('random'),cut=kwargs.get('cut')) # add sample with scaterer
        self.Green(E) #lead's Green's functions at E
        self.Dyson() #Dyson's equation
        
        self.D = np.imag(np.diagonal(self.G))
        ldos = []
        
        for i in np.arange(0,len(self.D),2):
            ldos.append(self.D[i]+self.D[i+1])
        
        ldos = np.array(np.round(ldos,6)).reshape((int(len(ldos)/self.L), self.L))
       
        return (-1*ldos/np.pi)
    
    
def plot_lead_spectrum(L, delta):
    '''Plot the spectrum of the wires as a function of their width and the onsite energy.'''
    QWZ = dirty_edge_QWZ_model(L, delta)
    
    k=np.linspace(-np.pi, np.pi,1000)
    S = QWZ.lead_spectrum(k)

    fig, ax = plt.subplots(1,1, figsize = (9,6))


    ax.plot(k,S, '--')

    ax.set_xticks(np.linspace(-np.pi,np.pi, 5))
    ax.set_xticklabels([r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'])

    ax.set_yticks(np.linspace(np.min(S), np.max(S), 5))
    ax.set_ylim(1.1*np.min(S),1.1*np.max(S))

    ax.tick_params(labelsize=20, direction='inout', size=10, width=2)

    ax.set_xlabel(r'$k$',fontsize=30)
    ax.set_ylabel(r'$E$',fontsize=30)

    ax.grid()

    plt.show()
    
I1 = interactive(plot_lead_spectrum,
                 L = IntSlider(value = 5, min = 5, max = 20, step = 5, description='$L$', continuous_update=False),
                 delta = FloatSlider(value = -1, min = -2, max = -1, step = 0.01, description='$\Delta$', continuous_update=False))


def plot_dens(n=10,delta=-1):
    '''plot particle particle density for different selfenergies'''
    QWZ = dirty_edge_QWZ_model(20,delta)
    QWZ.create_lead()
    QWZ.create_hamiltonian(20,100,(5,6))
    e,v = QWZ.spectrum_and_density()
    
    dens = np.log(np.sum((np.abs(np.asarray(np.abs(v)))**2)[:,n].reshape((int(QWZ.H.shape[0]/2),2)),axis=1))
    
 
    fig,axes = plt.subplots(1,2, figsize=(16,6.5))
    ax = axes[0]
    if QWZ.Vs > 0: 
        ax.plot(QWZ.V_x_coord,QWZ.V_y_coord, 'r-')
        N = matplotlib.colors.Normalize(np.min(dens[np.logical_not(QWZ.lattice_index)]), np.max(dens[np.logical_not(QWZ.lattice_index)]))
        colors = matplotlib.cm.viridis(N(dens))[np.logical_not(QWZ.lattice_index)]
        ax.scatter(QWZ.x[np.logical_not(QWZ.lattice_index)], QWZ.y[np.logical_not(QWZ.lattice_index)],s=100, c = colors)
        ax.plot(QWZ.x[QWZ.lattice_index], QWZ.y[QWZ.lattice_index], 'o', color='tab:red')
    else:
        N = matplotlib.colors.Normalize(np.min(dens), np.max(dens))
        colors = matplotlib.cm.viridis(N(dens))
        ax.scatter(QWZ.x, QWZ.y,s=100, c =colors)

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm = N, cmap=matplotlib.cm.viridis), ax=ax)
    cbar.ax.set_title('$\log(\left| \Psi \\right|^2)$')
    ax.tick_params(size=10, direction='inout', width=2, labelsize=20)
    
    ax = axes[1]
    
    index = np.arange(len(e)) + 1
    ax.plot(index, e, '--')
    ax.plot(index[n],e[n], 'ro', markersize=12)
    e_str = 'E = ' + str(np.round(e[n],4))
    ax.text(-2, 2, e_str, fontsize=20)
    
    ax.set_xlabel('#eigenstate', fontsize=30)
    ax.set_ylabel('E', fontsize=30)
    ax.set_ylim(-2.5,2.5)
    ax.tick_params(labelsize=10,size=10,width=2, direction='inout')
    ax.grid()
    plt.show()
    
I2 = interactive(plot_dens, 
                 n = IntSlider(value = 401, min = 1, max = 799, step = 1, description='n', continuous_update=False), 
                 delta = FloatSlider(value = -1, min = -2, max = -1, step = 0.01, description='$\Delta$', continuous_update=False))


def LDoS_ene_delta(ene=-0.5, delta=-1): 
    ''' LDoS as a function of energy and parameter delta'''
    ene=ene+1e-8*1j
    QWZ = dirty_edge_QWZ_model(L = 20, delta = delta)
    ldos = QWZ.LDOS(E = ene, V_size = (5,5), V_pos = (8,0), cut = True)
    ldos[1:-1,:][QWZ.lattice_index.reshape((QWZ.L, QWZ.L))]=0 

    fig,axes = plt.subplots(1,2, figsize=(2*8, 8*((QWZ.W+2)/QWZ.L)))
    
    ax=axes[0]
    ax.set_title('Cutout', fontsize = 25)
    im = ax.imshow(ldos, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad = 0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_title('LDoS', fontsize = 20)
    cbar.ax.tick_params(labelsize = 15) 
    ax.axis('off')
        
        
    QWZ = dirty_edge_QWZ_model(L = 20, delta = delta)
    ldos = QWZ.LDOS(E = ene, alpha = 10, V_size = (5,5), V_pos = (8,0))
    ldos[1:-1,:][QWZ.lattice_index.reshape((QWZ.L, QWZ.L))]=0
    
    ax=axes[1]
    ax.set_title('Random matrix', fontsize = 25)
    im = ax.imshow(ldos, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad = 0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_title('LDoS', fontsize = 20)
    cbar.ax.tick_params(labelsize = 15) 
    
    
    ax.axis('off')
    
    plt.show()
    
I3 = interactive(LDoS_ene_delta, 
                 ene = FloatSlider(value = -0.5, min = -4, max = 4, step = 0.01, description='E', continuous_update=False), 
                 delta = FloatSlider(value = -1, min = -3, max = -1, step = 0.01, description='$\Delta$', continuous_update=False))


def LDoS_y_l_w(y = 8, w = 5, l = 5): 
    ''' LDoS as a function of position and size of the dirt'''
    QWZ = dirty_edge_QWZ_model(L = 20, delta = -1)
    ldos = QWZ.LDOS(-0.5, V_size = (w,l), V_pos = (y,0), cut = True)
    try:
        ldos[1:-1,:][QWZ.lattice_index.reshape((QWZ.L, QWZ.L))]=0
    except:
        pass

    fig,axes = plt.subplots(1,2, figsize=(2*8, 8*((QWZ.W+2)/QWZ.L)))
    
    ax=axes[0]
    ax.set_title('Cutout', fontsize = 25)
    im = ax.imshow(ldos, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad = 0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_title('LDoS', fontsize = 20)
    cbar.ax.tick_params(labelsize = 15) 
    if l*w > 0:
        ax.text(0,y+l, '$L_s$', fontsize=20, color='white', rotation=90, va = 'top', ha = 'left')
        ax.text(w/2, y+l, '$W_s$', fontsize=20, color='white', va = 'top', ha = 'left')
    ax.axis('off')
        
        
    QWZ = dirty_edge_QWZ_model(L = 20, delta = -1)
    ldos = QWZ.LDOS(-0.5, alpha=10, V_size = (w,l), V_pos = (y,0))
    try:
        ldos[1:-1,:][QWZ.lattice_index.reshape((QWZ.L, QWZ.L))]=0
    except:
        pass
    ax=axes[1]
    ax.set_title('Random matrix', fontsize = 25)
    im = ax.imshow(ldos, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad = 0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_title('LDoS', fontsize = 20)
    cbar.ax.tick_params(labelsize = 15) 
    if l*w > 0:
        ax.text(0,y+l, '$L_s$', fontsize=20, color='white', rotation=90, va = 'top', ha = 'left')
        ax.text(w/2, y+l, '$W_s$', fontsize=20, color='white', va = 'top', ha = 'left')

    ax.axis('off')
    
    plt.show()
    
I4 = interactive(LDoS_y_l_w, 
                 y = IntSlider(value = 8, min = 0, max = 20, step = 1, description='$y$', continuous_update=False),
                 l = IntSlider(value = 2, min = 0, max = 20, step = 1, description='$L_s$', continuous_update=False),
                 w = IntSlider(value = 15, min = 0, max = 20, step = 1, description='$W_s$', continuous_update=False))


def LDoS_alpha(a): 
    ''' LDoS as a function of alpha'''
    QWZ = dirty_edge_QWZ_model(L = 20, delta = -1)
    ldos = QWZ.LDOS(-0.5, alpha=a, V_size = (15,3), V_pos = (8,0), random = True)
    if a > 0: ldos[1:-1,:][QWZ.lattice_index.reshape((QWZ.L, QWZ.L))]=0
    
    fig,ax = plt.subplots(1, 1, figsize=(2*8, 8*((QWZ.W+2)/QWZ.L)))
    im = ax.imshow(ldos, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad = 0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_title('LDoS', fontsize = 20)
    cbar.ax.tick_params(labelsize = 15) 
    ax.axis('off')
        
    plt.show()
    
I5 = interactive(LDoS_alpha, 
                 a = FloatSlider(value=0, min=0, max=50, step=0.25, description='$\\alpha$', continuous_update=False))
