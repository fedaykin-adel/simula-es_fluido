import matplotlib.pyplot as plt 
import numpy as np
import threading

def main():
    
    Nx = 400 
    Ny = 100
    rho0 = 100
    tau = 0.6
    Nt = 4000
    plotRealTime = True
    
    NL= 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36])
    
    F = np.ones((Ny,Nx,NL))
    np.random.seed(42)
    F += 0.01 * np.random.randn(Ny, Nx, NL)
    X,Y = np.meshgrid(range(Nx), range(Ny))
    F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))
    rho = np.sum(F,2)
    for i in idxs:
        F[:,:,i] *= rho0 / rho
    
    X, Y = np.meshgrid(range(Nx), range(Ny))
    cylinder = ( X - Nx/4) ** 2 + (Y - Ny/2)**2 < (Ny/4)**2
    
    fig = plt.figure(figsize=(4,2), dpi=80)
    
    for it in range(Nt):
        
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
        
        
        bndryF = F[cylinder,:]
        bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]   
        
        rho = np.sum(F,2)
        ux = np.sum(F*cxs, 2) / rho
        uy = np.sum(F*cys, 2) / rho
        
        

        
        Feq = np.zeros(F.shape)
        for i, cx,cy,w in zip(idxs,cxs, cys, weights):
            Feq[:,:,i] = rho * w * (1 + 3*(cx*ux+cy*uy) + 9*(cx*ux+cy*uy) **2/2 - 3*(ux**2+uy**2)/2)
        
            
        F += -(1.0/tau) * (F - Feq)
        F[cylinder,:]=bndryF
        
        if (plotRealTime and (it % 10)==0) or (it == Nt - 1):
            plt.cla()
            ux[cylinder] =0
            uy[cylinder] =0
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1,axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            vorticity[cylinder] = np.nan
            vorticity = np.ma.array(vorticity, mask=cylinder)
            plt.imshow(vorticity, cmap='bwr')
            plt.imshow(~cylinder, cmap='gray', alpha=0.3)
            plt.clim(-.1,.1)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            plt.pause(0.001)
            
    plt.savefig('blaa.png', dpi=240)
    plt.show()

def run_threads(num_threads, tg_thread):
	threads = []
	for i in range(num_threads):
		thread = threading.Thread(target=tg_thread)
		threads.append(thread)
		thread.start()
  
	for thread in threads:
		thread.join()

if __name__ == '__main__':
    num_thread = 10
    run_threads(num_thread,main)