import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper as jw
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import expm, expm_multiply, gmres
import itertools
import pyscf


class T_UPS():
    """ Tiled Unitary Product States
        Simplified for obtaining electron densities
    """
    def __init__(self, mol, Cmo, layers=1, include_doubles=True, 
    use_proj=True, pp=True, oo=True, include_dmat=False, plev=1, 
    include_first_singles=True, include_first_last_op=False, oo_layers=None):

        # define number of spin and spat orbitals
        self.no_spat = mol.nao
        self.no_spin = self.no_spat * 2
        # Num of alpha spin
        self.no_alpha = mol.nelec[0]
        # Num of beta spin
        self.no_beta = mol.nelec[1]
        # Basis size of Fock Space
        self.N = 2**self.no_spin

        self.perf_pair = pp
        self.orb_opt = oo
        self.S =  mol.intor('int1e_ovlp')
        if self.orb_opt and oo_layers is None:
            self.oo_layers = int(self.no_spat/2)
        else:
            self.oo_layers = oo_layers
        self.use_proj = use_proj

        if self.use_proj:
            self.initialise_projector()
        
        # include first singles in the first half layer of layer 1
        self.include_first_singles = include_first_singles
        self.include_first_last_op = include_first_last_op
        self.include_doubles = include_doubles

        # Number of operators (parameters)
        self.nop = self.no_spat - 1
        if(self.include_doubles): 
            self.nop *= 2
        
        self.initialise_tups_op_mat()
        if plev>0: print('Operator Matrices Generated')

        # operator order from left to right
        self.layers = layers
        self.initialise_op_order() 
        if plev>0: print('Operator Order Generated')

        self.initialise_ref()
        if plev>0: print('Wavefunction Reference Generated')

        self.Cmo = Cmo
        if self.perf_pair:
            self.Cmo = self.Cmo[:,[0,5,1,4,2,3]]
        
        self.hamiltonian_mo(mol)

        # self.include_dmat = include_dmat
        # if self.include_dmat:
            # self.initialise_spatial_rdm1_op()
            # self.initialise_spin_rdm1_op()
        # if plev>0: print('Density Matrix Operators Generated')
        
        # Current position
        self.x = np.zeros(self.dim)
        self.update()
        
    @property
    def dim(self):
        """Dimension of parameter vector, x"""
        return len(self.op_order)
        
    @property
    def energy(self):
        E = np.conj(self.wfn) @ self.mat_H @ self.wfn
        return E
    
    def take_step(self,step):
        """Take a step in parameter space"""
        self.x = np.mod(self.x + step + np.pi, 2*np.pi) - np.pi
        # self.x = np.mod(self.x + step + np.pi/2, np.pi) - np.pi/2
        self.update()
    
    def get_wfn(self,x):
        '''Rotates the wavefunction to generate a new wavefunction'''
        wfn = self.wf_ref.copy()
        for idx, op in enumerate(self.op_order):
            wfn = expm_multiply(self.kop_ij[op]*x[idx], wfn)
        return wfn

    def get_wfn_gradient(self,x):
        if self.use_proj:
            N = self.proj_N
        else:
            N = self.N

        if not self.approx_prec:
            wfn_grad = np.zeros((N, 2*self.dim))
        else:
            wfn_grad = np.zeros((N, self.dim))
        for j in range(wfn_grad.shape[1]):
            wfn_grad[:,j] = self.wf_ref.copy()
        
        for j, op in enumerate(self.op_order):
            wfn_grad = expm_multiply(self.kop_ij[op]*x[j], wfn_grad)
            wfn_grad[:,j] = self.kop_ij[op] @ wfn_grad[:,j]
            if not self.approx_prec:
                wfn_grad[:,j+self.dim] = self.kop_ij[op] @ wfn_grad[:,j]
        return wfn_grad[:,:self.dim], wfn_grad[:,self.dim:]

    def update(self):
        '''Updates the parameters'''
        self.wfn = self.get_wfn(self.x)
        # self.H_wfn = self.mat_H @ self.wfn
        # self.wfn_grad, self.wfn_hess = self.get_wfn_gradient(self.x)
        # print(self.x[6:12])
    
    def get_initial_guess(self):
        # Generate initial position
        # self.x = 2*np.pi*(np.random.rand(self.dim)-0.5)
        xstep = 0.1*(np.random.rand(self.dim)-0.5)
        self.take_step(xstep)
        self.update()
    
    def hamiltonian_mo(self, mol):
        '''Initialises the Hamiltonian matrix for given MOs'''
        # get integrals
        vnuc = mol.energy_nuc() # constant term
        
        oei = mol.intor('int1e_nuc') + mol.intor('int1e_kin')
        hpq  = np.linalg.multi_dot([self.Cmo.T, oei, self.Cmo]) # <p|h|q>

        eri = mol.intor("int2e", aosym="s1")
        eri_pqrs = pyscf.ao2mo.incore.full(eri, self.Cmo) # <pq|rs>
        eri_pqrs = eri_pqrs.transpose(0,2,1,3)

        H = FermionicOp({'':vnuc},num_spin_orbitals=self.no_spin)

        # one electron
        for q in range(self.no_spat):
            for p in range(self.no_spat):
                H += FermionicOp({f"+_{p} -_{q}": hpq[p,q]}, num_spin_orbitals=self.no_spin)
                H += FermionicOp({f"+_{p+self.no_spat} -_{q+self.no_spat}": hpq[p,q]}, num_spin_orbitals=self.no_spin)

        # two electron
        for s in range(self.no_spat):
            for r in range(self.no_spat):
                for q in range(self.no_spat):
                    for p in range(self.no_spat):
                        H += 0.5*FermionicOp({f"+_{p} +_{q+self.no_spat} -_{s+self.no_spat} -_{r}": eri_pqrs[p,q,r,s]}, num_spin_orbitals=self.no_spin)
                        H += 0.5*FermionicOp({f"+_{p+self.no_spat} +_{q} -_{s} -_{r+self.no_spat}": eri_pqrs[p,q,r,s]}, num_spin_orbitals=self.no_spin)
                        H += 0.5*FermionicOp({f"+_{p+self.no_spat} +_{q+self.no_spat} -_{s+self.no_spat} -_{r+self.no_spat}": eri_pqrs[p,q,r,s]}, num_spin_orbitals=self.no_spin)
                        H += 0.5*FermionicOp({f"+_{p} +_{q} -_{s} -_{r}": eri_pqrs[p,q,r,s]}, num_spin_orbitals=self.no_spin)
            
        # Save as a matrix
        self.mat_H = jw().map(H).to_matrix(sparse=True).real

        if self.use_proj:
            self.mat_H = csc_matrix(self.mat_proj.T @ (self.mat_H @ self.mat_proj))

    def initialise_ref(self):
        '''Initialises the HF reference state'''
        # create vacuum state
        wf_vac = np.zeros((self.N))
        wf_vac[0] = 1

        # create initial bitstring
        if self.perf_pair:
            strlst  = [f"+_{2*i}" for i in range(min(self.no_alpha,-(self.no_spat//-2)))] # -(self.no_spat//-2) does ceiling division
            strlst += [f"+_{2*i+1}" for i in range(min(self.no_alpha+(self.no_spat//-2),self.no_spat//2))] 
            strlst += [f"+_{2*i+self.no_spat}" for i in range(min(self.no_beta,-(self.no_spat//-2)))] 
            strlst += [f"+_{2*i+self.no_spat+1}" for i in range(min(self.no_beta+(self.no_spat//-2),self.no_spat//2))]
            # print(strlst)
        else:
            strlst = [f"+_{i}" for i in range(self.no_alpha)] + [f"+_{i+self.no_spat}" for i in range(self.no_beta)]

        init_op = FermionicOp({" ".join(strlst): 1.0}, num_spin_orbitals=self.no_spin)
        mat_init_op = jw().map(init_op).to_matrix(sparse=True).real
        self.wf_ref = mat_init_op @ wf_vac
        if self.use_proj:
            self.wf_ref = self.mat_proj.T @ self.wf_ref
    
    def initialise_op_mat(self):
        '''Initialise matrices using the 2nd quantised operators - general case'''
        self.kop_ij = {}
        # paired single
        count = 0
        for p in range(self.no_spat):
            for q in range(p):
                self.kop_ij[count] = self.get_singles_matrix(p,q)
                count += 1
        # paired doubles
        if(self.include_doubles):
            for p in range(self.no_spat):
                for q in range(p):
                    self.kop_ij[count] = self.get_doubles_matrix
                    count += 1
    
    def initialise_tups_op_mat(self):
        '''Initialise matrices using the 2nd quantised operators - tUPS case'''
        self.kop_ij = {}
        # paired single
        count = 0
        # defining k_10, k_32, k_54, ... k_pq. where q is even 
        for p in range(1, self.no_spat, 2):
            q = p-1
            self.kop_ij[count] = self.get_singles_matrix(p,q)
            count += 1

            # paired doubles
            if(self.include_doubles):
                self.kop_ij[count] = self.get_doubles_matrix(p,q)
                count += 1
        # defining k_21, k_43, k_65, ... k_pq. where q is odd 
        for q in range(1, self.no_spat-1, 2):
            p = q+1
            self.kop_ij[count] = self.get_singles_matrix(p,q)
            count += 1

            # paired doubles
            if(self.include_doubles):
                self.kop_ij[count] = self.get_doubles_matrix(p,q)
                count += 1
        
        if(self.include_first_last_op) and (self.no_spat % 2 == 0):
            self.kop_ij[count] = self.get_singles_matrix(0,self.no_spat-1)
            count += 1
        
            if(self.include_doubles):
                self.kop_ij[count] = self.get_doubles_matrix(0,self.no_spat-1)
                count += 1
        
    def initialise_op_order(self):
        self.op_order = []
        for i in range(0,len(self.kop_ij),2):
            self.op_order.extend([i,i+1,i])
        self.op_order.extend(self.op_order*(self.layers-1))

        if not self.include_first_singles:
            for i in range(int(self.no_spat/2)-1,-1,-1): # count down to 0
                self.op_order.pop(i*3)

        oo_order = []
        for i in range(0,len(self.kop_ij),2):
            oo_order.extend([i])
        if(self.orb_opt):
            self.op_order.extend(oo_order*self.oo_layers)

    def get_singles_matrix(self, p, q):
        t = FermionicOp({f"+_{p} -_{q}": 1}, num_spin_orbitals=self.no_spin)
        t += FermionicOp({f"+_{p+self.no_spat} -_{q+self.no_spat}": 1}, num_spin_orbitals=self.no_spin)
        k = t - t.adjoint()
        mat_k = jw().map(k).to_matrix(sparse=True).real
        if self.use_proj:
            mat_k = self.mat_proj.T @ (mat_k @ self.mat_proj) 
        return csc_matrix(mat_k)

    def get_doubles_matrix(self, p, q):
        t = FermionicOp({f"+_{p} +_{p+self.no_spat} -_{q+self.no_spat} -_{q}": 1.0}, num_spin_orbitals=self.no_spin)
        k = t - t.adjoint()
        mat_k = jw().map(k).to_matrix(sparse=True).real
        if self.use_proj:
            mat_k = self.mat_proj.T @ (mat_k @ self.mat_proj) 
        return csc_matrix(mat_k)

    def initialise_projector(self):
        # alpha spin combinations
        perm_alpha_str = '1'*self.no_alpha + '0'*(self.no_spat-self.no_alpha)
        alpha_perms = tuple(set(itertools.permutations(perm_alpha_str)))
        alpha_perms = (''.join(x) for x in alpha_perms)
        # beta spin combinations
        perm_beta_str = '1'*self.no_beta + '0'*(self.no_spat-self.no_beta)
        beta_perms = set(itertools.permutations(perm_beta_str))
        beta_perms = (''.join(x) for x in beta_perms)
        # get product of the 2 combinations
        full_perms = set(itertools.product(beta_perms,alpha_perms))
        proj_indices = []
        # convert indices from binary to decimal
        for x in full_perms:
            proj_indices.append(int(''.join(x),2))
        # sort indices from lowest to highest
        proj_indices.sort()
        # construct projector matrix and dimension of reduced space
        self.mat_proj = lil_matrix((self.N, len(proj_indices)))
        for j, idx in enumerate(proj_indices):
            self.mat_proj[idx,j] = 1
        self.mat_proj = self.mat_proj.tocsc()
        self.proj_N = len(proj_indices)
    
    def initialise_spatial_rdm2_op(self):
        if self.use_proj:
            self.doubly_rm_mat = np.zeros((self.no_spat,self.no_spat, self.N, self.proj_N))
        else:
            self.doubly_rm_mat = np.zeros((self.no_spat,self.no_spat, self.N, self.N))
        for r in range(self.no_spat):
            for s in range(self.no_spat):
                op = FermionicOp({f"-_{s+self.no_spat} -_{r}": 1.0}, num_spin_orbitals=self.no_spin)
                op += FermionicOp({f"-_{s} -_{r+self.no_spat}": 1.0}, num_spin_orbitals=self.no_spin)
                op += FermionicOp({f"-_{s+self.no_spat} -_{r+self.no_spat}": 1.0}, num_spin_orbitals=self.no_spin)
                op += FermionicOp({f"-_{s} -_{r}": 1.0}, num_spin_orbitals=self.no_spin)
                mat_op = jw().map(op).to_matrix(sparse=True).real
                if self.use_proj:
                    mat_op = mat_op @ self.mat_proj
                self.doubly_rm_mat[r,s,:,:] = mat_op.todense()
    
    def initialise_spatial_rdm1_op(self):
        if self.use_proj:
            self.singly_rm_mat_spat = np.zeros((self.no_spat,self.N,self.proj_N))
        else:
            self.singly_rm_mat_spat = np.zeros((self.no_spat,self.N,self.N))
        for r in range(self.no_spat):
            op = FermionicOp({f"-_{r}": 1.0}, num_spin_orbitals=self.no_spin)
            op += FermionicOp({f"-_{r+self.no_spat}": 1.0}, num_spin_orbitals=self.no_spin)
            mat_op = jw().map(op).to_matrix(sparse=True).real
            if self.use_proj:
                    mat_op = mat_op @ self.mat_proj
            self.singly_rm_mat_spat[r,:,:] = mat_op.todense()
    
    def initialise_spin_rdm1_op(self):
        if self.use_proj:
            self.singly_rm_mat_spin = np.zeros((self.no_spin,self.N,self.proj_N))
        else:
            self.singly_rm_mat_spin = np.zeros((self.no_spin,self.N,self.N))
        for r in range(self.no_spin):
            op = FermionicOp({f"-_{r}": 1.0}, num_spin_orbitals=self.no_spin)
            mat_op = jw().map(op).to_matrix(sparse=True).real
            if self.use_proj:
                    mat_op = mat_op @ self.mat_proj
            self.singly_rm_mat_spin[r,:,:] = mat_op.todense()
    
    # def spat_rdm1_mo(self):
    #     ket = self.singly_rm_mat_spat @ self.wfn
    #     density_mat = np.einsum('pi, ri->pr', ket, ket)
    #     return density_mat

    def spat_rdm1_mo(self):
        density_mat = np.zeros((self.no_spat,self.no_spat))
        for p in range(self.no_spat):
            for q in range(self.no_spat):
                op = FermionicOp({f"+_{q} -_{p}": 1.0}, num_spin_orbitals=self.no_spin)
                op += FermionicOp({f"+_{q+self.no_spat} -_{p+self.no_spat}": 1.0}, num_spin_orbitals=self.no_spin)
                mat_op = jw().map(op).to_matrix(sparse=True).real
                if self.use_proj:
                    mat_op = self.mat_proj.T @ mat_op @ self.mat_proj
                gamma_pq = self.wfn.T @ mat_op @ self.wfn
                density_mat[p,q] = gamma_pq
        return density_mat

    def spin_rdm1_mo(self):
        ket = self.singly_rm_mat_spin @ self.wfn
        density_mat = np.einsum('pi, ri->pr', ket, ket)
        return density_mat

    def spat_rdm1_ao(self):
        Dmo = self.spat_rdm1_mo()
        Dao = self.Cmo @ Dmo @ self.Cmo.T
        return Dao @ self.S
    
    def spin_rdm1_ao(self):
        Dmo = self.spin_rdm1_mo()
        Dmo_a = Dmo[:self.no_spat,:self.no_spat]
        Dmo_b = Dmo[self.no_spat:self.no_spin,self.no_spat:self.no_spin]
        Dao_a = self.Cmo @ Dmo_a @ self.Cmo.T
        Dao_b = self.Cmo @ Dmo_b @ self.Cmo.T
        return Dao_a @ self.S, Dao_b @ self.S

    def spat_rdm2_mo(self):
        ket = self.doubly_rm_mat @ self.wfn
        density_mat = np.einsum('pqi, rsi->pqrs', ket, ket)
        return density_mat
    
