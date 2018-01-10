import numpy as np

def nmf(V,b=0,K=32,n_iter=100):

    ''' NMF with multiplicative updates
 
     Input:
       - V: input positive matrix to be factorized (F x N)
       - b: 0 for IS, 1 for KL, 2 for Euc (beta divergence)
       - K: number of basis (NMF components)
       - n_iter: number of iterations

    
     Output:
       - W and H such that V=WH
       - cost: beta-divergence per iteration
    '''

    F,N = V.shape

    cost = np.zeros((n_iter,))
    
    #initialize W and H
    W = np.abs(np.random.randn(F,K))
    H = np.abs(np.random.randn(K,N))
    
    eps = 1e-20 #small value, avoid division by zero
    
    #compute current approximation
    V_ap = np.dot(W,H) #matrix multiplication
    Vtmp = V/V_ap #point-wise division
    if b==0:
        cost[0] = np.sum(Vtmp - np.log(Vtmp)) - F*N #IS divergence
    elif b==1:
        cost[0] = np.sum(V*np.log(Vtmp) - V + V_ap) #KL divergence
    else: #b==2
        cost[0] = np.linalg.norm(V-V_ap,'fro') #Euclidean

    
    de = 1 #corrective exponent
    if b==0:
        de = 0.5
    
    for itern in range(1,n_iter):
        
        #update W
        W =  W* (np.dot(V* np.power(V_ap,b-2),H.T) / np.dot(np.power(V_ap,b-1),H.T))**de
        V_ap = np.maximum(np.dot(W,H),eps)
    
        #update H
        H = H* (np.dot(W.T, V* np.power(V_ap,b-2)) / np.dot(W.T,np.power(V_ap,b-1)))**de
        V_ap = np.dot(W,H)
    
        #normalization
        scale = np.linalg.norm(W,axis=0) #norm of each column
        W = W/scale #normalize the columns
        H = H*scale[:,np.newaxis] #re-scale H

        Vtmp = V/V_ap #point-wise division
        if b==0:
            cost[itern] = np.sum(Vtmp - np.log(Vtmp)) - F*N #IS divergence
        elif b==1:
            cost[itern] = np.sum(V*np.log(Vtmp) - V + V_ap) #KL divergence
        else: #b==2
            cost[itern] = np.linalg.norm(V-V_ap,'fro') #Euclidean
    return W,H,cost


def l1_nmf(V,W,b=0,gam=1,n_iter=100):
    
    ''' NMF with l1 penalty
        
        Input:
        - V: input positive matrix to be factorized (F x N)
        - b: 0 for IS, 1 for KL, 2 for Euc
        - W: basis matrix
        - n_iter: number of iterations
        
        
        Output:
        - W and H such that V=WH
        - cost: beta-divergence per iteration
        '''
    
    F,N = V.shape
    K = W.shape[1]
    cost = np.zeros((n_iter,))
    
    #initialize H
    H = W.T.dot(V)
    
    eps = 1e-20 #small value, avoid division by zero
    
    #compute current approximation
    V_ap = np.dot(W,H) #matrix multiplication
    
    #compute initial cost
    Vtmp = V/V_ap #point-wise division
    if b==0:
        cost[0] = np.sum(Vtmp - np.log(Vtmp)) - F*N #IS divergence
    elif b==1:
        cost[0] = np.sum(V*np.log(Vtmp) - V + V_ap) #KL divergence
    else: #b==2
        cost[0] = np.linalg.norm(V-V_ap,'fro') #Euclidean

    de = 1 #corrective exponent
    if b==0:
        de = 0.5
    for itern in range(1,n_iter):
        
        #update H
        if b==2:
            tmp = np.maximum((np.dot(W.T, V* np.power(V_ap,b-2)) - gam) / (np.dot(W.T,np.power(V_ap,b-1))),eps)
            H = H* (tmp)**de
        else:
            H = H* (np.dot(W.T, V* np.power(V_ap,b-2)) / (np.dot(W.T,np.power(V_ap,b-1))+gam))**de
        V_ap = np.maximum(np.dot(W,H),eps)
        
        #current cost
        Vtmp = V/V_ap #point-wise division
        if b==0:
            cost[itern] = np.sum(Vtmp - np.log(Vtmp)) - F*N #IS divergence
        elif b==1:
            cost[itern] = np.sum(V*np.log(Vtmp) - V + V_ap) #KL divergence
        else: #b==2
            cost[itern] = np.linalg.norm(V-V_ap,'fro') #Euclidean
    return H,cost

def log_nmf(V,W,p,b=0,lam=1,n_iter=100):
    
    ''' NMF with group sparsity
        
        Input:
        - V: input positive matrix to be factorized (F x N)
        - b: 0 for IS, 1 for KL, 2 for Euc
        - W: basis matrix
        - n_iter: number of iterations
        - p: group sizes
        
        
        Output:
        - W and H such that V=WH
        - cost: beta-divergence per iteration
        '''
    
    F,N = V.shape
    K = W.shape[1]
    cost = np.zeros((n_iter,))
    D = p.shape[0] #number of groups
    k = p[0] #number of rows per group (assuming equally sized groups)
    
    #initialize H
    H = W.T.dot(V)
    
    eps = 1e-20 #small value, avoid division by zero
    
    #compute current approximation
    V_ap = np.dot(W,H) #matrix multiplication
    
    #compute initial cost
    Vtmp = V/V_ap #point-wise division
    if b==0:
        cost[0] = np.sum(Vtmp - np.log(Vtmp)) - F*N #IS divergence
    elif b==1:
        cost[0] = np.sum(V*np.log(Vtmp) - V + V_ap) #KL divergence
    else: #b==2
        cost[0] = np.linalg.norm(V-V_ap,'fro') #Euclidean

    de = 1 #corrective exponent
    if b==0:
        de = 0.5
    for itern in range(1,n_iter):
        
        #update H
        #update H
        group_norms = np.linalg.norm(np.reshape(H.T,(k*N,D),order='F'),axis=0,ord=1) #D groups
        
        #row_norm = np.linalg.norm(H,ord=1,axis=1)
        P = np.repeat(1./(eps+group_norms),k)
        
        if b==2:
            tmp = np.maximum((np.dot(W.T, V* np.power(V_ap,b-2)) - lam*P[:,np.newaxis]) / (np.dot(W.T,np.power(V_ap,b-1))),eps)
            H = H* (tmp)**de
        else:
            H = H* (np.dot(W.T, V* np.power(V_ap,b-2)) / (np.dot(W.T,np.power(V_ap,b-1))+lam*P[:,np.newaxis]))**de
        V_ap = np.maximum(np.dot(W,H),eps)
        
        # compute current cost value
        Vtmp = V/V_ap #point-wise division
        if b==0:
            cost[itern] = np.sum(Vtmp - np.log(Vtmp)) - F*N #IS divergence
        elif b==1:
            cost[itern] = np.sum(V*np.log(Vtmp) - V + V_ap) #KL divergence
        else: #b==2
            cost[itern] = np.linalg.norm(V-V_ap,'fro') #Euclidean
    return H,cost

def log_l1_nmf(V,W,p,b=0,lam=1,gam=1,n_iter=100):
    
    ''' NMF with group sparsity and l1 penalty
        
        Input:
        - V: input positive matrix to be factorized (F x N)
        - b: 0 for IS, 1 for KL, 2 for Euc
        - W: basis matrix
        - n_iter: number of iterations
        - lam: group sparsity parameter
        - gam: sparsity parameter
        - p: group sizes
        
        
        Output:
        - W and H such that V=WH
        - cost: beta-divergence per iteration
        '''
    
    F,N = V.shape
    K = W.shape[1]
    cost = np.zeros((n_iter,))
    D = p.shape[0] #number of groups
    k = p[0] #number of rows per group (assuming equally sized groups)
    
    #initialize H
    H = W.T.dot(V)
    
    eps = 1e-20 #small value, avoid division by zero
    
    #compute current approximation
    V_ap = np.dot(W,H) #matrix multiplication
    
    #compute initial cost
    Vtmp = V/V_ap #point-wise division
    if b==0:
        cost[0] = np.sum(Vtmp - np.log(Vtmp)) - F*N #IS divergence
    elif b==1:
        cost[0] = np.sum(V*np.log(Vtmp) - V + V_ap) #KL divergence
    else: #b==2
        cost[0] = np.linalg.norm(V-V_ap,'fro') #Euclidean

    de = 1 #corrective exponent
    if b==0:
        de = 0.5
    for itern in range(1,n_iter):
        
        #update H
        group_norms = np.linalg.norm(np.reshape(H.T,(k*N,D),order='F'),axis=0,ord=1) #D groups
        
        #row_norm = np.linalg.norm(H,ord=1,axis=1)
        P = np.repeat(1./(eps+group_norms),k)
        if b==2:
            tmp = np.maximum((np.dot(W.T, V* np.power(V_ap,b-2)) - gam - lam*P[:,np.newaxis]) / (np.dot(W.T,np.power(V_ap,b-1))),eps)
            H = H* (tmp)**de
        else:
            H = H* (np.dot(W.T, V* np.power(V_ap,b-2)) / (np.dot(W.T,np.power(V_ap,b-1))+lam*P[:,np.newaxis]+gam))**de
        V_ap = np.maximum(np.dot(W,H),eps)
        
        # compute current cost value (IS divergence)
        Vtmp = V/V_ap #point-wise division
        if b==0:
            cost[itern] = np.sum(Vtmp - np.log(Vtmp)) - F*N #IS divergence
        elif b==1:
            cost[itern] = np.sum(V*np.log(Vtmp) - V + V_ap) #KL divergence
        else: #b==2
            cost[itern] = np.linalg.norm(V-V_ap,'fro') #Euclidean


    return H,cost