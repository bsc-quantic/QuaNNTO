#!/usr/bin/env python3
"""Audit all trained coordinates of the three odd-like cat checkpoints.
Run: OPENBLAS_NUM_THREADS=1 python catstate_full_parameter_audit.py --repo .
Uses the repository's independent parameter_fock_reference, not JAX.
Writes a separate output directory; never overwrites trained checkpoints.
Finite-Fock results require cutoff convergence; Hessian eigenvalues near zero
are not a proof of a strict minimum. Uses the original 15-moment raw SSE.
"""
from pathlib import Path
import argparse, importlib.util, json, time
import numpy as np
from scipy.optimize import minimize

CONFIGS = [(1, [[0]]), (2, [[0, 1]]), (1, [[0], [0]])]

def observables(c):
    a=np.diag(np.sqrt(np.arange(1,c)),1);ad=a.T;n=ad@a
    pw=lambda k:np.linalg.matrix_power(a,k)
    return np.array([a,pw(2),n,pw(3),pw(2)@ad,n@n,pw(4),pw(3)@ad,pw(5),pw(4)@ad,pw(3)@ad@ad,n@n@n,pw(6),pw(5)@ad,pw(4)@ad@ad])

def cat(c,odd=False):
    v=np.ones(c)
    for n in range(1,c):v[n]=v[n-1]/np.sqrt(n)
    v[(np.arange(c)%2)!=(1 if odd else 0)]=0
    return v/np.linalg.norm(v)

class Objective:
    def __init__(self,engine,N,subs,target,cutoff):
        self.engine,self.N,self.subs,self.target,self.c=engine,N,subs,target,cutoff
        self.ops=observables(cutoff);self.plus=cat(cutoff);self.minus=cat(cutoff,True)
    def state(self,p):
        v=self.engine(p,self.N,self.subs,-1,self.c).reshape(self.c,-1)
        norm=float(np.vdot(v,v).real)
        if norm<1e-28:raise ValueError('Vanishing branch norm')
        v=v/np.sqrt(norm);return v@v.conj().T,norm
    def residual(self,p):
        rho,_=self.state(p)
        delta=np.einsum('ij,kji->k',rho,self.ops)-self.target
        return np.r_[delta.real,delta.imag]
    def __call__(self,p):
        r=self.residual(p);return float(r@r)
    def metrics(self,p):
        rho,norm=self.state(p)
        return dict(loss=self(p),F_plus=float(np.vdot(self.plus,rho@self.plus).real),F_minus=float(np.vdot(self.minus,rho@self.minus).real),parity=float(np.dot((-1.)**np.arange(self.c),rho.diagonal().real)),purity=float(np.trace(rho@rho).real),branch_norm=norm,output_tail_last10=float(rho.diagonal().real[-10:].sum()))

def derivatives(f,x,h):
    n=len(x);eye=np.eye(n)*h;f0=f(x);H=np.empty((n,n));g=np.empty(n)
    for i in range(n):
        fp,fm=f(x+eye[i]),f(x-eye[i]);g[i]=(fp-fm)/(2*h);H[i,i]=(fp-2*f0+fm)/h**2
        for j in range(i):
            H[i,j]=H[j,i]=(f(x+eye[i]+eye[j])-f(x+eye[i]-eye[j])-f(x-eye[i]+eye[j])+f(x-eye[i]-eye[j]))/(4*h*h)
    vals,vecs=np.linalg.eigh(H)
    return dict(step=h,gradient=g.tolist(),gradient_norm=float(np.linalg.norm(g)),eigenvalues=vals.tolist(),eigenvectors_columns=vecs.tolist())

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',type=Path,default=Path('.'));ap.add_argument('--out',type=Path,default=Path('catstate_full_parameter_audit'));ap.add_argument('--refine',action='store_true');args=ap.parse_args()
    root=args.repo.resolve();args.out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('parameter_fock_reference',root/'quannto/utils/strawberryfields/parameter_fock_reference.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    target=np.load(root/'data/datasets/catstate_phi0.0_trainsize1_stats15_cut20_rng-1to1_outputs.npy').reshape(-1)
    results=[]
    for N,subs in CONFIGS:
        L=len(subs);stem=f'catstate_phi0.0_trainsize1_stats15_cut20_rng-1.0to-1.0_N{N}_L{L}_sub{subs}_inNone'
        p=np.load(root/'results/models/params'/f'{stem}.npy');record=json.loads((root/'results/models/training_records'/f'{stem}.json').read_text())
        print('\nSTART',stem,flush=True);row=dict(stem=stem,N=N,subtractions=subs,parameters=p.tolist(),training_selection=record['selection'],bounds=record['bounds'])
        row['cutoff_checks']={str(c):Objective(mod.parameter_fock_ket,N,subs,target,c).metrics(p) for c in (40,60,80)}
        f=Objective(mod.parameter_fock_ket,N,subs,target,60)
        print('metrics',row['cutoff_checks'],flush=True)
        row['derivatives']=[derivatives(f,p,h) for h in (2e-4,5e-5)]
        for d in row['derivatives']:print('h',d['step'],'grad',d['gradient_norm'],'eig',d['eigenvalues'],flush=True)
        def matrix(d):
            V=np.array(d['eigenvectors_columns']);return (V*np.array(d['eigenvalues']))@V.T
        large,small=row['derivatives']
        # Step ratio 4: cancel the leading O(h^2) finite-difference error.
        HR=(16*matrix(small)-matrix(large))/15
        row['richardson_eigenvalues']=np.linalg.eigvalsh(HR).tolist()
        row['richardson_gradient']=((16*np.array(small['gradient'])-np.array(large['gradient']))/15).tolist()
        # Follow lowest-curvature direction to distinguish real descent from FD noise.
        d=row['derivatives'][-1];v=np.array(d['eigenvectors_columns'])[:,0]
        row['lowest_eigenvector_scan']=[dict(t=t,**f.metrics(p+t*v)) for t in (-.03,-.01,-.003,-.001,0,.001,.003,.01,.03)]
        if args.refine:
            def grad(x):
                h=2e-6;eye=np.eye(len(x))*h
                return np.array([(f(x+e)-f(x-e))/(2*h) for e in eye])
            starts=[('checkpoint',p)]
            if d['eigenvalues'][0]<-1e-3:starts += [('minus',p-.03*v),('plus',p+.03*v)]
            row['refinements']=[]
            for name,x in starts:
                res=minimize(f,x,jac=grad,method='L-BFGS-B',bounds=record['bounds'],options=dict(maxiter=600,ftol=1e-14,gtol=1e-7,maxls=40))
                rr=dict(start=name,success=bool(res.success),message=str(res.message),nit=int(res.nit),parameters=res.x.tolist(),metrics60=f.metrics(res.x),metrics80=Objective(mod.parameter_fock_ket,N,subs,target,80).metrics(res.x))
                row['refinements'].append(rr);print('refine',name,rr,flush=True)
        results.append(row);(args.out/'audit.json').write_text(json.dumps(results,indent=2))
    print('Saved',args.out/'audit.json',flush=True)
if __name__=='__main__':main()
