
import numpy as np
from ecos import solve
from scipy.sparse import coo_matrix , csc_matrix , issparse

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# idLogit: driver routine
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def idLogit( K , I , N , y , X , ind , constant=False , Lambdas=[1000.0,1000.0] , prints={} , **kwargs ) : 
    
    """idLogit model estimation routine, MLE problem solved with the ECOS solver. 

    Calls specialized routines based on specific penalization. 

    Args: 
        K (int): the number of model features
        I (int): the number of individuals for which there is data
        N (int): the total number of observations
        y (numpy.array): A length-N vector of choices, coded as +/- 1
        X (numpy.array or scipy.sparse): A N x K
        ind (list): A length-N list of individual indices (1,...,I) for each observation
        constant (:obj:`bool`, optional): include a constant in the model (true), or don't (false)
        Lambdas (:obj:`list`, optional): L1 and L2 penalty weights, both default to 1000
        prints (:obj:`dict`, optional): List of extra setup prints to do
        **kwargs: Keyword arguments passed directly to ecos.solve

    Returns:
        x (numpy.array): A length K (or K+1) array of estimated coefficients
    
    """
    
    if( Lambdas[0] <= 0.0 ) : # no L1 penalty
        if( Lambdas[1] <= 0.0 ) : # niether penalties
            return idLogit_np( K , I , N , y , X , ind , constant=constant , prints=prints , **kwargs )
        else : # L2 penalty only
            return idLogit_l2( K , I , N , y , X , ind , constant=constant , Lambda2=Lambda[1] , prints=prints , **kwargs )
    else : # Lambdas[0] > 0
        if( Lambdas[1] <= 0.0 ) : # L1 penalty only
            return idLogit_l1( K , I , N , y , X , ind , constant=constant , Lambda1=Lambdas[0] , prints=prints , **kwargs )
        else : # both penalties
            return idLogit_en( K , I , N , y , X , ind , constant=constant , Lambdas=Lambdas , prints=prints , **kwargs )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# 
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def idLogit_np( K , I , N , y , X , ind , constant=False , prints={} , **kwargs ) : 
    
    """idLogit with no penalties. 

    Technically, a _very_ underdetermined problem, and the coefficients are trivial. 

    Args: 
        K (int): the number of model features
        I (int): the number of individuals for which there is data
        N (int): the total number of observations
        y (numpy.array): A length-N vector of choices, coded as +/- 1
        X (numpy.array or scipy.sparse): A N x K
        ind (list): A length-N list of individual indices (1,...,I) for each observation
        constant (:obj:`bool`, optional): include a constant in the model (true), or don't (false)
        Lambda1 (:obj:`float`, optional): L1 penalty weight, defaults to 1000
        prints (:obj:`dict`, optional): List of extra setup prints to do
        **kwargs: Keyword arguments passed directly to ecos.solve

    Returns:
        x (numpy.array): A length K (or K+1) array of estimated coefficients
    
    """
    
    return 
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# 
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def idLogit_l1( K , I , N , y , X , ind , constant=False , Lambda1=1000.0 , prints={} , **kwargs ) : 
    
    """idLogit model estimation with L1 penalty, MLE solved with the ECOS solver. 
    
    Args: 
        K (int): the number of model features
        I (int): the number of individuals for which there is data
        N (int): the total number of observations
        y (numpy.array): A length-N vector of choices, coded as +/- 1
        X (numpy.array or scipy.sparse): A N x K
        ind (list): A length-N list of individual indices (1,...,I) for each observation
        constant (:obj:`bool`, optional): include a constant in the model (true), or don't (false)
        Lambda1 (:obj:`float`, optional): L1 penalty weight, defaults to 1000
        prints (:obj:`dict`, optional): List of extra setup prints to do
        **kwargs: Keyword arguments passed directly to ecos.solve

    Returns:
        x (numpy.array): A length K (or K+1) array of estimated coefficients

    """

    if( Lambda1 <= 0.0 ) : 
        return idLogit_np( K , I , N , y , X )

    IK = I * K
    Ninv  = 1.0 / N
    if issparse(X) : 
        X = X.tocoo()
        Xnnz = X.nnz 
    else : 
        Xnnz = X.shape[0] * X.shape[1]


    Nvars = K + 3 * IK + 4 * N          # b , l , u , v , w , d , p , m
    
    Ncons = N + K + IK                  # u + v + w ; d(1) + ... + d(I) ; d - p + m
    Annz  = 3 * N + 4 * IK              # 3N ; IK ; 3IK
    
    Ncone = N + 2 * IK + 6 * N          # w, p, m in Pos ; Exp variables
    Gnnz  = 5 * N + 2 * IK + 2 * Xnnz   # N + 2IK ; 4N + 2Xnnz
    
    # with these sizes, we can estimate the memory requirements...
    
    
    # convenience list that lets us easily index sparse matrix terms
    indices = np.arange( 0 , max(IK,2*N) , dtype=np.int )

    # convenience values of "variable" starts and lengths in a 
    # concatenated Nvars-vector of all variables. 
    starts , length = {} , {}
    starts['b'] , length['b'] = 0 , K
    starts['l'] , length['l'] = starts['b'] + length['b'] , N
    starts['u'] , length['u'] = starts['l'] + length['l'] , N
    starts['v'] , length['v'] = starts['u'] + length['u'] , N
    starts['w'] , length['w'] = starts['v'] + length['v'] , N
    starts['d'] , length['d'] = starts['w'] + length['w'] , IK
    starts['p'] , length['p'] = starts['d'] + length['d'] , IK
    starts['m'] , length['m'] = starts['p'] + length['p'] , IK
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # COST VECTOR (ie, objective)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    c = np.zeros( Nvars , dtype=np.float_ )
    c[ starts['l'] : starts['l'] + length['l'] ] = Ninv
    c[ starts['p'] : starts['p'] + length['p'] ] = Lambdas[0] * Ninv
    c[ starts['m'] : starts['m'] + length['m'] ] = Lambdas[0] * Ninv
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # LINEAR EQUALITY CONSTRAINTS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # 
    #     u + v + w = 1           N rows     3N nonzeros (each of N rows has 3 terms)
    #     d(1) + ... + d(I) = 0   K rows     IK nonzeros (each of K rows has I terms)
    #     d - p + m = 0          IK rows    3IK nonzeros (each of IK rows has 3 terms)
    # 
    
    Arows = np.zeros( Annz , dtype=np.int )
    Acols = np.zeros( Annz , dtype=np.int )
    Adata = np.ones( Annz , dtype=np.float_ ) # almost all of the constraint data terms are "1"
    
    j , jj = 0 , 0
    
    # u + v + w
    jj = j + N
    Arows[j:jj] = indices[0:N]
    Acols[j:jj] = starts['u'] + indices[0:N]
    j = jj
    
    jj = j + N
    Arows[j:jj] = indices[0:N]
    Acols[j:jj] = starts['v'] + indices[0:N]
    j = jj
    
    jj = j + N
    Arows[j:jj] = indices[0:N]
    Acols[j:jj] = starts['w'] + indices[0:N]
    j = jj
    
    # d(1) + ... + d(I), stored in d "K-first"
    for k in range(0,K) : 
        jj = j + I
        Arows[j:jj] = N + k 
        Acols[j:jj] = starts['d'] + k + K * indices[0:I] 
        j = jj

    # d - p + m, noting that we have to set data for "p" terms as well as rows/cols
    jj = j + IK
    Arows[j:jj] = N+K+indices[0:IK]
    Acols[j:jj] = starts['d'] + indices[0:IK]
    j = jj
    
    jj = j + IK
    Arows[j:jj] = N+K+indices[0:IK] 
    Acols[j:jj] = starts['p'] + indices[0:IK]
    Adata[j:jj] = -1.0
    j = jj
    
    jj = j + IK
    Arows[j:jj] = N+K+indices[0:IK]
    Acols[j:jj] = starts['m'] + indices[0:IK]
    j = jj
    
    A = csc_matrix( (Adata,(Arows,Acols)) , shape=(Ncons,Nvars) , dtype=np.float_ );

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # LINEAR EQUALITY CONSTRAINT RHS (forecably initialize fewer terms)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if( N < IK + K ) : 
        b = np.zeros( Ncons , dtype=np.float_ )
        b[0:N] = 1.0;
    else : 
        b = np.ones( Ncons , dtype=np.float_ )
        b[N:] = 0.0;
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DIMENSIONS OF CONIC CONSTRAINTS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    dims = { 
        'l' : N + 2*IK , # w, p, and m must be non-negative
        'q' : [ 1+IK ] , # (t,d) lie in the second-order cone
        'e' : 2*N        # 2 triplets of Exp cone variables for each n (3N "variables")
    }
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CONIC CONSTRAINT RHS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    h = np.zeros( Ncone , dtype=np.float_ )
    h[ dims['l'] + dims['q'][0] + 3*indices[0:2*N] + 2 ] = 1.0
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CONIC CONSTRAINTS MATRIX
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    # The first N+2IK rows of G are easily described: 
    # 
    #    b  l  u  v  w  d  p  m  t
    # 
    #    0  0  0  0 -I  0  0  0  0   N rows,  N non-zeros
    #    0  0  0  0  0  0 -I  0  0  IK rows, IK non-zeros
    #    0  0  0  0  0  0  0 -I  0  IK rows, IK non-zeros
    # 
    # This suggests initializing Gdata entries to -1 and filling in Grows
    # and Gcols accordingly. 
    # 
    # The remainder are not so easily described, but are very similar to 
    # cases reviewed above. Particularly, for n = 1,...,N
    # 
    #    G[ N+2IK + 3n - 1 , : ] = G[ 1+N+3IK + 6n - 1 , : ] = 0'
    # 
    # are empty rows, and we don't have to do anything. For n = 0,...,N-1
    # 
    #    G[ N+2IK + 3n + 0 , starts['l'] + n ] =  1
    #    G[ N+2IK + 3n + 1 , starts['u'] + n ] = -1
    #    G[ N+2IK + 3n + 3 , starts['l'] + n ] =  1
    #    G[ N+2IK + 3n + 4 , starts['v'] + n ] = -1
    # 
    # this is thus 4N non-zeros. Only b and d terms remain: 
    # 
    #    G[ N+2IK + 3n + 0 , 0:K ] = y[n] X[:,n]'
    #    G[ N+2IK + 3n + 0 , start[n] : start[n] + K ] = y[n] X[:,n]'
    # 
    # where 
    # 
    #    start[n] = starts['d'] + K*(i(n)-1) 
    # 
    # There are thus 2 "Xnnz" non-zeros here, where Xnnz is the number
    # of non-zeros in the X matrix. This is the only part that requires changing 
    # data. 
    
    Grows = np.zeros( Gnnz , dtype=np.int )
    Gcols = np.zeros( Gnnz , dtype=np.int )
    Gdata = - np.ones( Gnnz , dtype=np.float_ )
    
    j , jj , base = 0 , 0 , 0
    
    # w, p, m terms are non-negative
    
    jj = j +  N
    Grows[j:jj] = base + indices[0: N] 
    Gcols[j:jj] = starts['w'] + indices[0:length['w']]
    j = jj
    base +=  N
    
    jj = j + IK
    Grows[j:jj] = base + indices[0:IK] 
    Gcols[j:jj] = starts['p'] + indices[0:length['p']]
    j = jj
    base += IK
    
    jj = j + IK
    Grows[j:jj] = base + indices[0:IK]
    Gcols[j:jj] = starts['m'] + indices[0:length['m']]
    j = jj
    base += IK
    
    # base is fixed now, because we intersperse the exponential cone terms 
    
    # u, v terms in Exp
    jj = j + N
    Grows[j:jj] = base + 6*indices[0:N] + 1
    Gcols[j:jj] = starts['u'] + indices[0:length['u']]
    j = jj
    
    jj = j + N
    Grows[j:jj] = base + 6*indices[0:N] + 4
    Gcols[j:jj] = starts['v'] + indices[0:length['v']]
    j = jj
    
    # l terms in Exp
    jj = j + N
    Grows[j:jj] = base + 6*indices[0:N] + 0
    Gcols[j:jj] = starts['l'] + indices[0:length['l']]
    Gdata[j:jj] = 1.0
    j = jj
    
    jj = j + N
    Grows[j:jj] = base + 6*indices[0:N] + 3
    Gcols[j:jj] = starts['l'] + indices[0:length['l']]
    Gdata[j:jj] = 1.0
    j = jj
    
    # b, d terms in Exp ** hardest part ** handle differently for sparse and dense X
    if( issparse(X) ) : 
        
        jj = j + X.nnz
        Grows[j:jj] = base + 6 * X.row
        Gcols[j:jj] = starts['b'] + X.col
        Gdata[j:jj] = X.data
        j = jj
        
        jj = j + X.nnz
        Grows[j:jj] = base + 6 * X.row
        Gcols[j:jj] = starts['d'] + length['b'] * (ind[X.row]-1) + X.col
        Gdata[j:jj] = X.data
        j = jj
        
    else : 
        
        for n in range(0,N) : 

            data = - y[n] * X[n,:]

            jj = j + length['b']
            Grows[j:jj] = base + 6*n
            Gcols[j:jj] = starts['b'] + indices[0:length['b']]
            Gdata[j:jj] = data
            j = jj

            jj = j + length['b']
            Grows[j:jj] = base + 6*n
            Gcols[j:jj] = starts['d'] + length['b'] * (ind[n]-1) + indices[0:length['b']]
            Gdata[j:jj] = data
            j = jj
    
    G = csc_matrix( (Gdata,(Grows,Gcols)) , shape=(Ncone,Nvars) , dtype=np.float_ )
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # OPTIONAL PRINTS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if 'start' in prints and prints['start'] : 
        print( "\nVariable Starts: \n" )
        print( starts )

    if 'costs' in prints and prints['costs'] : 

        print( "\nCosts: \n" )

        for k in starts : 
            if( np.max( np.abs( c[starts[k]:starts[k]+length[k]] ) ) == 0 ) : 
                pass # print( "variable: %s is zero" % k )
            else :
                print( "variable: %s" % k )
                print( c[starts[k]:starts[k]+length[k]] )
    
    if 'lineq' in prints and prints['lineq'] : 
        
        print( "\nLinear Equality Constraints: \n" )
        
        Array = A.toarray()
        
        blocks = [ N , K , IK ]
        
        base , baseB = 0 , 0
        for B in blocks : 
            baseB = base + B
            print( "\nA: %i to %i\n" %(base,baseB) )
            rows = np.arange(base,baseB)
            for k in starts : 
                if( np.max( np.max( np.abs( Array[rows,starts[k]:starts[k]+length[k]] ) ) ) == 0 ) : 
                    pass # print( "variable: %s is zero" % k )
                else :
                    print( "variable: %s" % k )
                    print( Array[rows,starts[k]:starts[k]+length[k]] )
            base = baseB
            
        del Array
        
    if 'lerhs' in prints and prints['lerhs'] : 
        
        print( "\nLinear Equality Constraints RHS: \n" )
        
        blocks = [ N , K , IK ]
        
        base , baseB = 0 , 0
        for B in blocks : 
            baseB = base + B
            print( b[base:baseB] )
            base = baseB

    if 'cones' in prints and prints['cones'] : 
        
        print( "\nConic Constraints: \n" )
        
        Grray = G.toarray()
        
        blocks = [ N , IK , IK , 6*N ]
        
        base , baseB = 0 , 0
        for B in blocks : 
            baseB = base + B
            print( "\nG: [%i,%i)\n" %(base,baseB) )
            rows = np.arange(base,baseB)
            for k in starts : 
                if( np.max( np.max( np.abs( Grray[rows,starts[k]:starts[k]+length[k]] ) ) ) == 0 ) : 
                    pass # print( "variable: %s is zero" % k )
                else :
                    print( "variable: %s" % k )
                    print( Grray[rows,starts[k]:starts[k]+length[k]] )
            base = baseB
            
        del Grray
    
    if 'ccrhs' in prints and prints['ccrhs'] : 
        
        print( "\nConic Constraints RHS: \n" )
        
        blocks = [ N , IK , IK , 6*N ]
        
        base , baseB = 0 , 0
        for B in blocks : 
            baseB = base + B
            print( h[base:baseB] )
            base = baseB
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CLEANUP
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            
    del indices
    del Arows , Acols , Adata
    del Grows , Gcols , Gdata
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # SOLVE ATTEMPT
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    res = solve( c , G , h , dims , A , b , **kwargs )
    return res['x'][0:K] , res['info']
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# 
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def idLogit_l2( K , I , N , y , X , ind , constant=False , Lambda2=1000.0 , prints={} , **kwargs ) : 
    
    """idLogit model estimation with L2 penalty, MLE solved with the ECOS solver. 
    
    Args: 
        K (int): the number of model features
        I (int): the number of individuals for which there is data
        N (int): the total number of observations
        y (numpy.array): A length-N vector of choices, coded as +/- 1
        X (numpy.array or scipy.sparse): A N x K
        ind (list): A length-N list of individual indices (1,...,I) for each observation
        constant (:obj:`bool`, optional): include a constant in the model (true), or don't (false)
        Lambda1 (:obj:`float`, optional): L1 penalty weight, defaults to 1000
        prints (:obj:`dict`, optional): List of extra setup prints to do
        **kwargs: Keyword arguments passed directly to ecos.solve

    Returns:
        x (numpy.array): A length K (or K+1) array of estimated coefficients

    """

    if( Lambda2 <= 0.0 ) : 
        return idLogit_np( K , I , N , y , X )


    IK = I * K
    Ninv  = 1.0 / N
    if issparse(X) : 
        X = X.tocoo()
        Xnnz = X.nnz 
    else : 
        Xnnz = X.shape[0] * X.shape[1]

    Nvars = K + IK + 4 * N + 1              # b , l , u , v , w , d , t
    
    Ncons = N + K                           # u + v + w ; d(1) + ... + d(I)
    Annz  = 3 * N + IK                      # 3N ; IK
    
    Ncone = N + 1 + IK + 6 * N              # w in Pos ; (t,d) in SOC ; Exp vars
    Gnnz  = 1 + 5 * N + IK + 2 * Xnnz       # N ; 1 + IK ; 4N + 2Xnnz
    
    # with these sizes, we can estimate the memory requirements...
    
    
    # convenience list that lets us easily index sparse matrix terms
    indices = np.arange( 0 , max(IK,2*N) , dtype=np.int )

    # convenience values of "variable" starts and lengths in a 
    # concatenated Nvars-vector of all variables. 
    starts , length = {} , {}
    starts['b'] , length['b'] = 0 , K
    starts['l'] , length['l'] = starts['b'] + length['b'] , N
    starts['u'] , length['u'] = starts['l'] + length['l'] , N
    starts['v'] , length['v'] = starts['u'] + length['u'] , N
    starts['w'] , length['w'] = starts['v'] + length['v'] , N
    starts['d'] , length['d'] = starts['w'] + length['w'] , IK
    starts['t'] , length['t'] = starts['d'] + length['d'] , 1
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # COST VECTOR (ie, objective)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    c = np.zeros( Nvars , dtype=np.float_ )
    c[ starts['l'] : starts['l'] + length['l'] ] = Ninv
    c[ starts['t'] ] = Lambdas[1] * Ninv / 2.0
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # LINEAR EQUALITY CONSTRAINTS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # 
    #     u + v + w = 1           N rows     3N nonzeros (each of N rows has 3 terms)
    #     d(1) + ... + d(I) = 0   K rows     IK nonzeros (each of K rows has I terms)
    # 
    
    Arows = np.zeros( Annz , dtype=np.int )
    Acols = np.zeros( Annz , dtype=np.int )
    Adata = np.ones( Annz , dtype=np.float_ ) # almost all of the constraint data terms are "1"
    
    j , jj = 0 , 0
    
    # u + v + w
    jj = j + N
    Arows[j:jj] = indices[0:N]
    Acols[j:jj] = starts['u'] + indices[0:N]
    j = jj
    
    jj = j + N
    Arows[j:jj] = indices[0:N]
    Acols[j:jj] = starts['v'] + indices[0:N]
    j = jj
    
    jj = j + N
    Arows[j:jj] = indices[0:N]
    Acols[j:jj] = starts['w'] + indices[0:N]
    j = jj
    
    # d(1) + ... + d(I), stored in d "K-first"
    for k in range(0,K) : 
        jj = j + I
        Arows[j:jj] = N + k 
        Acols[j:jj] = starts['d'] + k + K * indices[0:I] 
        j = jj
    
    A = csc_matrix( (Adata,(Arows,Acols)) , shape=(Ncons,Nvars) , dtype=np.float_ );

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # LINEAR EQUALITY CONSTRAINT RHS (presume N > K)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    b = np.zeros( Ncons , dtype=np.float_ )
    b[0:N] = 1.0;
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DIMENSIONS OF CONIC CONSTRAINTS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    dims = { 
        'l' : N ,        # w must be non-negative
        'q' : [ 1+IK ] , # (t,d) lie in the second-order cone
        'e' : 2*N        # 2 triplets of Exp cone variables for each n (3N "variables")
    }
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CONIC CONSTRAINT RHS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    h = np.zeros( Ncone , dtype=np.float_ )
    h[ dims['l'] + dims['q'][0] + 3*indices[0:2*N] + 2 ] = 1.0
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CONIC CONSTRAINTS MATRIX
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    # The first 1+N+IK rows of G are easily described: 
    # 
    #    b  l  u  v  w  d  p  m  t
    # 
    #    0  0  0  0 -I  0  0  0  0   N rows,  N non-zeros
    #    0  0  0  0  0  0  0  0 -1   1 row,   1 non-zero
    #    0  0  0  0  0 -I  0  0  0  IK rows, IK non-zeros
    # 
    # This suggests initializing Gdata entries to -1 and filling in Grows
    # and Gcols accordingly. 
    # 
    # The remainder are not so easily described, but are very similar to 
    # cases reviewed above. Particularly, for n = 1,...,N
    # 
    #    G[ 1+N+3IK + 3n - 1 , : ] = G[ 1+N+3IK + 6n - 1 , : ] = 0'
    # 
    # are empty rows, and we don't have to do anything. For n = 0,...,N-1
    # 
    #    G[ 1+N+IK + 3n + 0 , starts['l'] + n ] =  1
    #    G[ 1+N+IK + 3n + 1 , starts['u'] + n ] = -1
    #    G[ 1+N+IK + 3n + 3 , starts['l'] + n ] =  1
    #    G[ 1+N+IK + 3n + 4 , starts['v'] + n ] = -1
    # 
    # this is thus 4N non-zeros. Only b and d terms remain: 
    # 
    #    G[ 1+N+IK + 3n + 0 , 0:K ] = y[n] X[:,n]'
    #    G[ 1+N+IK + 3n + 0 , start[n] : start[n] + K ] = y[n] X[:,n]'
    # 
    # where 
    # 
    #    start[n] = starts['d'] + K*(i(n)-1) 
    # 
    # There are thus 2 "Xnnz" non-zeros here, where Xnnz is the number
    # of non-zeros in the X matrix. This is the only part that requires changing 
    # data. 
    
    Grows = np.zeros( Gnnz , dtype=np.int )
    Gcols = np.zeros( Gnnz , dtype=np.int )
    Gdata = - np.ones( Gnnz , dtype=np.float_ )
    
    j , jj , base = 0 , 0 , 0
    
    # w, terms are non-negative
    
    jj = j +  N
    Grows[j:jj] = base + indices[0: N] 
    Gcols[j:jj] = starts['w'] + indices[0:length['w']]
    j = jj
    base +=  N
    
    # t and d terms in the SOC
    
    Grows[j] , Gcols[j] = base , starts['t'] ; j += 1
    base += 1
    
    jj = j + IK
    Grows[j:jj] = base + indices[0:IK] 
    Gcols[j:jj] = starts['d'] + indices[0:length['d']]
    j = jj
    base += IK
    
    # base is fixed now, because we intersperse the exponential cone terms 
    
    # u, v terms in Exp
    jj = j + N
    Grows[j:jj] = base + 6*indices[0:N] + 1
    Gcols[j:jj] = starts['u'] + indices[0:length['u']]
    j = jj
    
    jj = j + N
    Grows[j:jj] = base + 6*indices[0:N] + 4
    Gcols[j:jj] = starts['v'] + indices[0:length['v']]
    j = jj
    
    # l terms in Exp
    jj = j + N
    Grows[j:jj] = base + 6*indices[0:N] + 0
    Gcols[j:jj] = starts['l'] + indices[0:length['l']]
    Gdata[j:jj] = 1.0
    j = jj
    
    jj = j + N
    Grows[j:jj] = base + 6*indices[0:N] + 3
    Gcols[j:jj] = starts['l'] + indices[0:length['l']]
    Gdata[j:jj] = 1.0
    j = jj
    
    # b, d terms in Exp ** hardest part ** handle differently for sparse and dense X
    if( issparse(X) ) : 
        
        jj = j + X.nnz
        Grows[j:jj] = base + 6 * X.row
        Gcols[j:jj] = starts['b'] + X.col
        Gdata[j:jj] = X.data
        j = jj
        
        jj = j + X.nnz
        Grows[j:jj] = base + 6 * X.row
        Gcols[j:jj] = starts['d'] + length['b'] * (ind[X.row]-1) + X.col
        Gdata[j:jj] = X.data
        j = jj
        
    else : 
        
        for n in range(0,N) : 

            data = - y[n] * X[n,:]

            jj = j + length['b']
            Grows[j:jj] = base + 6*n
            Gcols[j:jj] = starts['b'] + indices[0:length['b']]
            Gdata[j:jj] = data
            j = jj

            jj = j + length['b']
            Grows[j:jj] = base + 6*n
            Gcols[j:jj] = starts['d'] + length['b'] * (ind[n]-1) + indices[0:length['b']]
            Gdata[j:jj] = data
            j = jj
    
    G = csc_matrix( (Gdata,(Grows,Gcols)) , shape=(Ncone,Nvars) , dtype=np.float_ )
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # OPTIONAL PRINTS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if 'start' in prints and prints['start'] : 
        print( "\nVariable Starts: \n" )
        print( starts )

    if 'costs' in prints and prints['costs'] : 

        print( "\nCosts: \n" )

        for k in starts : 
            if( np.max( np.abs( c[starts[k]:starts[k]+length[k]] ) ) == 0 ) : 
                pass # print( "variable: %s is zero" % k )
            else :
                print( "variable: %s" % k )
                print( c[starts[k]:starts[k]+length[k]] )
    
    if 'lineq' in prints and prints['lineq'] : 
        
        print( "\nLinear Equality Constraints: \n" )
        
        Array = A.toarray()
        
        blocks = [ N , K ]
        
        base , baseB = 0 , 0
        for B in blocks : 
            baseB = base + B
            print( "\nA: %i to %i\n" %(base,baseB) )
            rows = np.arange(base,baseB)
            for k in starts : 
                if( np.max( np.max( np.abs( Array[rows,starts[k]:starts[k]+length[k]] ) ) ) == 0 ) : 
                    pass # print( "variable: %s is zero" % k )
                else :
                    print( "variable: %s" % k )
                    print( Array[rows,starts[k]:starts[k]+length[k]] )
            base = baseB
            
        del Array
        
    if 'lerhs' in prints and prints['lerhs'] : 
        
        print( "\nLinear Equality Constraints RHS: \n" )
        
        blocks = [ N , K ]
        
        base , baseB = 0 , 0
        for B in blocks : 
            baseB = base + B
            print( b[base:baseB] )
            base = baseB

    if 'cones' in prints and prints['cones'] : 
        
        print( "\nConic Constraints: \n" )
        
        Grray = G.toarray()
        
        blocks = [ N , 1+IK , 6*N ]
        
        base , baseB = 0 , 0
        for B in blocks : 
            baseB = base + B
            print( "\nG: [%i,%i)\n" %(base,baseB) )
            rows = np.arange(base,baseB)
            for k in starts : 
                if( np.max( np.max( np.abs( Grray[rows,starts[k]:starts[k]+length[k]] ) ) ) == 0 ) : 
                    pass # print( "variable: %s is zero" % k )
                else :
                    print( "variable: %s" % k )
                    print( Grray[rows,starts[k]:starts[k]+length[k]] )
            base = baseB
            
        del Grray
    
    if 'ccrhs' in prints and prints['ccrhs'] : 
        
        print( "\nConic Constraints RHS: \n" )
        
        blocks = [ N , 1+IK , 6*N ]
        
        base , baseB = 0 , 0
        for B in blocks : 
            baseB = base + B
            print( h[base:baseB] )
            base = baseB
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CLEANUP
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            
    del indices
    del Arows , Acols , Adata
    del Grows , Gcols , Gdata
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # SOLVE ATTEMPT
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    res = solve( c , G , h , dims , A , b , **kwargs )
    return res['x'][0:K] , res['info']
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# 
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
def idLogit_en( K , I , N , y , X , ind , constant=False , Lambdas=[1000.0,1000.0] , prints={} , **kwargs ) : 
    
    """idLogit model estimation with Elastic Net penalty, MLE solved with the ECOS solver. 
    
    Args: 
        K (int): the number of model features
        I (int): the number of individuals for which there is data
        N (int): the total number of observations
        y (numpy.array): A length-N vector of choices, coded as +/- 1
        X (numpy.array or scipy.sparse): A N x K
        ind (list): A length-N list of individual indices (1,...,I) for each observation
        constant (:obj:`bool`, optional): include a constant in the model (true), or don't (false)
        Lambda1 (:obj:`float`, optional): L1 penalty weight, defaults to 1000
        prints (:obj:`dict`, optional): List of extra setup prints to do
        **kwargs: Keyword arguments passed directly to ecos.solve

    Returns:
        x (numpy.array): A length K (or K+1) array of estimated coefficients

    """

    IK = I * K
    Ninv  = 1.0 / N
    if issparse(X) : 
        X = X.tocoo()
        Xnnz = X.nnz 
    else : 
        Xnnz = X.shape[0] * X.shape[1]

    Nvars = K + 3 * IK + 4 * N + 1          # b, l, u, v, w, d, p, m, t
    
    Ncons = N + K + IK                      # u + v + w ; d(1) + ... + d(I) ; d - p + m
    Annz  = 3 * N + 4 * IK                  # 3N ; IK ; 3IK
    
    Ncone = N + 1 + 3 * IK + 6 * N          # w, p, m in Pos ; (t,d) in SOC ; Exp vars
    Gnnz  = 1 + 5 * N + 3 * IK + 2 * Xnnz   # N + 2IK ; 1 + IK ; 4N + 2Xnnz
    
    # with these sizes, we can estimate the memory requirements...
    
    
    # convenience list that lets us easily index sparse matrix terms
    indices = np.arange( 0 , max(IK,2*N) , dtype=np.int )

    # convenience values of "variable" starts and lengths in a 
    # concatenated Nvars-vector of all variables. 
    starts , length = {} , {}
    starts['b'] , length['b'] = 0 , K
    starts['l'] , length['l'] = starts['b'] + length['b'] , N
    starts['u'] , length['u'] = starts['l'] + length['l'] , N
    starts['v'] , length['v'] = starts['u'] + length['u'] , N
    starts['w'] , length['w'] = starts['v'] + length['v'] , N
    starts['d'] , length['d'] = starts['w'] + length['w'] , IK
    starts['p'] , length['p'] = starts['d'] + length['d'] , IK
    starts['m'] , length['m'] = starts['p'] + length['p'] , IK
    starts['t'] , length['t'] = starts['m'] + length['m'] , 1
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # COST VECTOR (ie, objective)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    c = np.zeros( Nvars , dtype=np.float_ )
    c[ starts['l'] : starts['l'] + length['l'] ] = Ninv
    c[ starts['p'] : starts['p'] + length['p'] ] = Lambdas[0] * Ninv
    c[ starts['m'] : starts['m'] + length['m'] ] = Lambdas[0] * Ninv
    c[ starts['t'] ] = Lambdas[1] * Ninv / 2.0
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # LINEAR EQUALITY CONSTRAINTS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # 
    #     u + v + w = 1           N rows     3N nonzeros (each of N rows has 3 terms)
    #     d(1) + ... + d(I) = 0   K rows     IK nonzeros (each of K rows has I terms)
    #     d - p + m = 0          IK rows    3IK nonzeros (each of IK rows has 3 terms)
    # 
    
    Arows = np.zeros( Annz , dtype=np.int )
    Acols = np.zeros( Annz , dtype=np.int )
    Adata = np.ones( Annz , dtype=np.float_ ) # almost all of the constraint data terms are "1"
    
    j , jj = 0 , 0
    
    # u + v + w
    jj = j + N
    Arows[j:jj] = indices[0:N]
    Acols[j:jj] = starts['u'] + indices[0:N]
    j = jj
    
    jj = j + N
    Arows[j:jj] = indices[0:N]
    Acols[j:jj] = starts['v'] + indices[0:N]
    j = jj
    
    jj = j + N
    Arows[j:jj] = indices[0:N]
    Acols[j:jj] = starts['w'] + indices[0:N]
    j = jj
    
    # d(1) + ... + d(I), stored in d "K-first"
    for k in range(0,K) : 
        jj = j + I
        Arows[j:jj] = N + k 
        Acols[j:jj] = starts['d'] + k + K * indices[0:I] 
        j = jj

    # d - p + m, noting that we have to set data for "p" terms as well as rows/cols
    jj = j + IK
    Arows[j:jj] = N+K+indices[0:IK]
    Acols[j:jj] = starts['d'] + indices[0:IK]
    j = jj
    
    jj = j + IK
    Arows[j:jj] = N+K+indices[0:IK] 
    Acols[j:jj] = starts['p'] + indices[0:IK]
    Adata[j:jj] = -1.0
    j = jj
    
    jj = j + IK
    Arows[j:jj] = N+K+indices[0:IK]
    Acols[j:jj] = starts['m'] + indices[0:IK]
    j = jj
    
    A = csc_matrix( (Adata,(Arows,Acols)) , shape=(Ncons,Nvars) , dtype=np.float_ );

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # LINEAR EQUALITY CONSTRAINT RHS (forecably initialize fewer terms)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if( N < IK + K ) : 
        b = np.zeros( Ncons , dtype=np.float_ )
        b[0:N] = 1.0;
    else : 
        b = np.ones( Ncons , dtype=np.float_ )
        b[N:] = 0.0;
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DIMENSIONS OF CONIC CONSTRAINTS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    dims = { 
        'l' : N + 2*IK , # w, p, and m must be non-negative
        'q' : [ 1+IK ] , # (t,d) lie in the second-order cone
        'e' : 2*N        # 2 triplets of Exp cone variables for each n (3N "variables")
    }
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CONIC CONSTRAINT RHS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    h = np.zeros( Ncone , dtype=np.float_ )
    h[ dims['l'] + dims['q'][0] + 3*indices[0:2*N] + 2 ] = 1.0
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CONIC CONSTRAINTS MATRIX
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    # The first 1+N+3IK rows of G are easily described: 
    # 
    #    b  l  u  v  w  d  p  m  t
    # 
    #    0  0  0  0 -I  0  0  0  0   N rows,  N non-zeros
    #    0  0  0  0  0  0 -I  0  0  IK rows, IK non-zeros
    #    0  0  0  0  0  0  0 -I  0  IK rows, IK non-zeros
    #    0  0  0  0  0  0  0  0 -1   1 row,   1 non-zero
    #    0  0  0  0  0 -I  0  0  0  IK rows, IK non-zeros
    # 
    # This suggests initializing Gdata entries to -1 and filling in Grows
    # and Gcols accordingly. 
    # 
    # The remainder are not so easily described, but are very similar to 
    # cases reviewed above. Particularly, for n = 1,...,N
    # 
    #    G[ 1+N+3IK + 3n - 1 , : ] = G[ 1+N+3IK + 6n - 1 , : ] = 0'
    # 
    # are empty rows, and we don't have to do anything. For n = 0,...,N-1
    # 
    #    G[ 1+N+3IK + 3n + 0 , starts['l'] + n ] =  1
    #    G[ 1+N+3IK + 3n + 1 , starts['u'] + n ] = -1
    #    G[ 1+N+3IK + 3n + 3 , starts['l'] + n ] =  1
    #    G[ 1+N+3IK + 3n + 4 , starts['v'] + n ] = -1
    # 
    # this is thus 4N non-zeros. Only b and d terms remain: 
    # 
    #    G[ 1+N+3IK + 3n + 0 , 0:K ] = y[n] X[:,n]'
    #    G[ 1+N+3IK + 3n + 0 , start[n] : start[n] + K ] = y[n] X[:,n]'
    # 
    # where 
    # 
    #    start[n] = starts['d'] + K*(i(n)-1) 
    # 
    # There are thus 2 "Xnnz" non-zeros here, where Xnnz is the number
    # of non-zeros in the X matrix. This is the only part that requires changing 
    # data. 
    
    Grows = np.zeros( Gnnz , dtype=np.int )
    Gcols = np.zeros( Gnnz , dtype=np.int )
    Gdata = - np.ones( Gnnz , dtype=np.float_ )
    
    j , jj , base = 0 , 0 , 0
    
    # w, p, m terms are non-negative
    
    jj = j +  N
    Grows[j:jj] = base + indices[0: N] 
    Gcols[j:jj] = starts['w'] + indices[0:length['w']]
    j = jj
    base +=  N
    
    jj = j + IK
    Grows[j:jj] = base + indices[0:IK] 
    Gcols[j:jj] = starts['p'] + indices[0:length['p']]
    j = jj
    base += IK
    
    jj = j + IK
    Grows[j:jj] = base + indices[0:IK]
    Gcols[j:jj] = starts['m'] + indices[0:length['m']]
    j = jj
    base += IK
    
    # t and d terms in the SOC
    
    Grows[j] , Gcols[j] = base , starts['t'] ; j += 1
    base += 1
    
    jj = j + IK
    Grows[j:jj] = base + indices[0:IK] 
    Gcols[j:jj] = starts['d'] + indices[0:length['d']]
    j = jj
    base += IK
    
    # base is fixed now, because we intersperse the exponential cone terms 
    
    # u, v terms in Exp
    jj = j + N
    Grows[j:jj] = base + 6*indices[0:N] + 1
    Gcols[j:jj] = starts['u'] + indices[0:length['u']]
    j = jj
    
    jj = j + N
    Grows[j:jj] = base + 6*indices[0:N] + 4
    Gcols[j:jj] = starts['v'] + indices[0:length['v']]
    j = jj
    
    # l terms in Exp
    jj = j + N
    Grows[j:jj] = base + 6*indices[0:N] + 0
    Gcols[j:jj] = starts['l'] + indices[0:length['l']]
    Gdata[j:jj] = 1.0
    j = jj
    
    jj = j + N
    Grows[j:jj] = base + 6*indices[0:N] + 3
    Gcols[j:jj] = starts['l'] + indices[0:length['l']]
    Gdata[j:jj] = 1.0
    j = jj
    
    # b, d terms in Exp ** hardest part ** handle differently for sparse and dense X
    if( issparse(X) ) : 
        
        jj = j + X.nnz
        Grows[j:jj] = base + 6 * X.row
        Gcols[j:jj] = starts['b'] + X.col
        Gdata[j:jj] = X.data
        j = jj
        
        jj = j + X.nnz
        Grows[j:jj] = base + 6 * X.row
        Gcols[j:jj] = starts['d'] + length['b'] * (ind[X.row]-1) + X.col
        Gdata[j:jj] = X.data
        j = jj
        
    else : 
        
        for n in range(0,N) : 

            data = - y[n] * X[n,:]

            jj = j + length['b']
            Grows[j:jj] = base + 6*n
            Gcols[j:jj] = starts['b'] + indices[0:length['b']]
            Gdata[j:jj] = data
            j = jj

            jj = j + length['b']
            Grows[j:jj] = base + 6*n
            Gcols[j:jj] = starts['d'] + length['b'] * (ind[n]-1) + indices[0:length['b']]
            Gdata[j:jj] = data
            j = jj
    
    G = csc_matrix( (Gdata,(Grows,Gcols)) , shape=(Ncone,Nvars) , dtype=np.float_ )
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # OPTIONAL PRINTS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    if 'start' in prints and prints['start'] : 
        print( "\nVariable Starts: \n" )
        print( starts )

    if 'costs' in prints and prints['costs'] : 

        print( "\nCosts: \n" )

        for k in starts : 
            if( np.max( np.abs( c[starts[k]:starts[k]+length[k]] ) ) == 0 ) : 
                pass # print( "variable: %s is zero" % k )
            else :
                print( "variable: %s" % k )
                print( c[starts[k]:starts[k]+length[k]] )
    
    if 'lineq' in prints and prints['lineq'] : 
        
        print( "\nLinear Equality Constraints: \n" )
        
        Array = A.toarray()
        
        blocks = [ N , K , IK ]
        
        base , baseB = 0 , 0
        for B in blocks : 
            baseB = base + B
            print( "\nA: %i to %i\n" %(base,baseB) )
            rows = np.arange(base,baseB)
            for k in starts : 
                if( np.max( np.max( np.abs( Array[rows,starts[k]:starts[k]+length[k]] ) ) ) == 0 ) : 
                    pass # print( "variable: %s is zero" % k )
                else :
                    print( "variable: %s" % k )
                    print( Array[rows,starts[k]:starts[k]+length[k]] )
            base = baseB
            
        del Array
        
    if 'lerhs' in prints and prints['lerhs'] : 
        
        print( "\nLinear Equality Constraints RHS: \n" )
        
        blocks = [ N , K , IK ]
        
        base , baseB = 0 , 0
        for B in blocks : 
            baseB = base + B
            print( b[base:baseB] )
            base = baseB

    if 'cones' in prints and prints['cones'] : 
        
        print( "\nConic Constraints: \n" )
        
        Grray = G.toarray()
        
        blocks = [ N , IK , IK , 1+IK , 6*N ]
        
        base , baseB = 0 , 0
        for B in blocks : 
            baseB = base + B
            print( "\nG: [%i,%i)\n" %(base,baseB) )
            rows = np.arange(base,baseB)
            for k in starts : 
                if( np.max( np.max( np.abs( Grray[rows,starts[k]:starts[k]+length[k]] ) ) ) == 0 ) : 
                    pass # print( "variable: %s is zero" % k )
                else :
                    print( "variable: %s" % k )
                    print( Grray[rows,starts[k]:starts[k]+length[k]] )
            base = baseB
            
        del Grray
    
    if 'ccrhs' in prints and prints['ccrhs'] : 
        
        print( "\nConic Constraints RHS: \n" )
        
        blocks = [ N , IK , IK , 1+IK , 6*N ]
        
        base , baseB = 0 , 0
        for B in blocks : 
            baseB = base + B
            print( h[base:baseB] )
            base = baseB
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CLEANUP
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            
    del indices
    del Arows , Acols , Adata
    del Grows , Gcols , Gdata
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # SOLVE ATTEMPT
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    res = solve( c , G , h , dims , A , b , **kwargs )
    return res['x'][0:K] , res['info']
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 
# Copyright W. Ross Morrow 2018
# 
# morrowwr@gmail.com, wrossmorrow.com
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
