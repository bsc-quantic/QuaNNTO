"""Remove the redundant single-mode circuit coordinate and check all 4 physical directions.
Run from the repo root with the audit script in the same directory as this file.
"""
import sys,json,argparse
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from quannto.tasks.diagnostics.catstate_full_parameter_audit import observables,cat,derivatives

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path('.'));ap.add_argument('--out',type=Path,default=Path('catstate_full_parameter_audit'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    c=70;a=np.diag(np.sqrt(np.arange(1,c)),1);ad=a.T;ops=observables(c)
    target=np.load(args.repo/'data/datasets/catstate_phi0.0_trainsize1_stats15_cut20_rng-1to1_outputs.npy').reshape(-1)
    def objective(x):
        br,bi,r,theta=x
        v=np.zeros(c,complex);v[0]=1/np.sqrt(np.cosh(r))
        for n in range(0,c-2,2):v[n+2]=v[n]*np.tanh(r)*np.sqrt((n+1)*(n+2))/(n+2)
        v*=np.exp(-1j*theta*np.arange(c))
        b=br+1j*bi
        v=a@(expm(b*ad-b.conjugate()*a)@v);v/=np.linalg.norm(v)
        delta=np.einsum('i,kij,j->k',v.conj(),ops,v)-target
        return float(np.vdot(delta,delta).real)
    r=minimize_scalar(lambda r:objective([0,0,r,0]),bounds=(.2,.3),method='bounded',options={'xatol':1e-14}).x
    x=np.array([0,0,r,0]);d1=derivatives(objective,x,2e-4);d2=derivatives(objective,x,5e-5)
    def hm(d):v=np.array(d['eigenvectors_columns']);return (v*np.array(d['eigenvalues']))@v.T
    H=(16*hm(d2)-hm(d1))/15
    j=dict(coordinates=['Re(beta)','Im(beta)','r_positive','theta_rotation'],state='normalized a D(beta) R(theta) S(-r)|0>',r=r,loss=objective(x),eigenvalues_Richardson=np.linalg.eigvalsh(H).tolist(),derivatives=[d1,d2])
    (args.out/'single_mode_physical.json').write_text(json.dumps(j,indent=2));print(j)

if __name__ == "__main__":
    main()
