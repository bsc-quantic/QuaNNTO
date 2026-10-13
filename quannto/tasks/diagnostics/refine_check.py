import sys,importlib.util,json,argparse
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).parent));from quannto.tasks.diagnostics.catstate_full_parameter_audit import Objective,derivatives

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path('.'));ap.add_argument('--out',type=Path,default=Path('catstate_full_parameter_audit'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    root=args.repo;spec=importlib.util.spec_from_file_location('ref',root/'quannto/utils/strawberryfields/parameter_fock_reference.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    target=np.load(root/'data/datasets/catstate_phi0.0_trainsize1_stats15_cut20_rng-1to1_outputs.npy').reshape(-1)
    stem='catstate_phi0.0_trainsize1_stats15_cut20_rng-1.0to-1.0_N1_L2_sub[[0], [0]]_inNone'
    p=np.load(root/'results/models/params'/f'{stem}.npy');f=Objective(m.parameter_fock_ket,1,[[0],[0]],target,60)
    lo=np.full(10,-np.inf);hi=-lo;lo[[2,7]]=-np.log(5.5);hi[[2,7]]=np.log(5.5)
    res=least_squares(f.residual,p,jac='3-point',bounds=(lo,hi),ftol=3e-14,xtol=3e-14,gtol=3e-10,max_nfev=400,x_scale='jac',verbose=0)
    print('LS',res.message,res.nfev,np.linalg.norm(res.grad),f.metrics(res.x),res.x,flush=True)
    d1=derivatives(f,res.x,1e-4);d2=derivatives(f,res.x,5e-5)
    def hm(d):v=np.array(d['eigenvectors_columns']);return (v*np.array(d['eigenvalues']))@v.T
    H=(4*hm(d2)-hm(d1))/3;ev,V=np.linalg.eigh(H);print('Richardson eig',ev,flush=True)
    row=dict(parameters=res.x.tolist(),metrics60=f.metrics(res.x),metrics80=Objective(m.parameter_fock_ket,1,[[0],[0]],target,80).metrics(res.x),nfev=res.nfev,optimality=res.optimality,gradient=res.grad.tolist(),message=res.message,eigenvalues_Richardson=ev.tolist(),hessians=[d1,d2])
    (args.out/'two_layer_refinement.json').write_text(json.dumps(row,indent=2))

if __name__ == "__main__":
    main()
