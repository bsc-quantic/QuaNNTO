"""Validate the eight 150926 cat results and export paper-ready LaTeX tables.

Run quantum_tables first (see docs/catstate_results_150926.md), then:
python -m quannto.tasks.catstate_report --evaluation catstate_results
No optimization or model deserialization is performed.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from quannto.utils.path_utils import quannto_results_dir
from .quantum_tables import load_numeric, _atomic_text


ORIGINAL_IDS = ('cat_n1_double', 'cat_n2_aux_double', 'cat_n2_both',
                'cat_n3_aux_pair', 'cat_n2_l2')
EXTRA_IDS = ('cat_n1_single', 'cat_n1_triple', 'cat_n1_l2')


def _serialized_array(value):
    """Read only raw bytes of this campaign's JAX jsonpickle array format.

    Never execute py/function, py/reduce or constructors from the JSON.
    The producer uses little-endian float64/complex128; check byte counts.
    """
    state = value['py/reduce'][1]['py/tuple'][2]['py/tuple']
    version, shape, _, fortran, data = state
    if version != 1 or fortran or not isinstance(data, dict) or 'py/b64' not in data:
        raise ValueError('Unsupported serialized array representation')
    shape = tuple(shape['py/tuple'])
    raw = base64.b64decode(data['py/b64'], validate=True)
    count = int(np.prod(shape))
    if len(raw) == count*8:
        dtype = '<f8'
    elif len(raw) == count*16:
        dtype = '<c16'
    else:
        raise ValueError('Serialized array byte count mismatch')
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _fidelity_text(value):
    return r'$<10^{-10}$' if value < 1e-10 else f'{value:.6f}'


def make_table(rows, label):
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Cat-state preparation for the even target '
             r'$|C_+\rangle=(|1\rangle+|-1\rangle)/\sqrt{2(1+e^{-2})}$, '
             r'where the kets denote coherent states. Each model is selected '
             r'by the minimum recorded training loss in one run with seed 42 '
             r'and four basinhopping proposals. The moment loss and the squared '
             r'fidelity $F=\langle C_+|\rho_1|C_+\rangle$ refer to the same '
             r'checkpoint; $F_{\rm init}$ is evaluated at the exact random '
             r'parameter vector from which that run was initialized, before any '
             r'optimization. $\rho_1$ is the normalized reduced state of the first '
             r'output mode for input $\alpha=-1$. Subtraction lists use one-based '
             r'mode indices, retain repeated events, and are separated by '
             r'semicolons between layers.}',
             r'\label{'+label+'}', r'\begin{tabular}{rrlrrr}', r'\hline',
             r'$N$ & $L$ & Subtractions & $\mathcal L_{\min}$ & $F_{\rm init}$ & $F$ \\', r'\hline']
    for r in rows:
        k = ';'.join('['+','.join(str(i+1) for i in layer)+']' for layer in r['subtractions'])
        lines.append(f"{r['N']} & {r['L']} & $({k})$ & {r['loss_history_min']:.6g} & "
                     +_fidelity_text(r['initial_fidelity_mean'])+' & '
                     +_fidelity_text(r['fidelity_mean'])+r' \\')
    lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--models-root', type=Path,
                        default=quannto_results_dir()/'models')
    args = parser.parse_args(argv)
    report = json.loads((args.evaluation/'quantum_tables.json').read_text())
    by_id = {r['id']:r for r in report['models'] if r['task']=='cat'}
    if set(by_id) != set(ORIGINAL_IDS+EXTRA_IDS):
        raise ValueError('This report requires all eight explicitly mapped cat configurations')
    rows = [by_id[i] for i in ORIGINAL_IDS+EXTRA_IDS]
    checks = []
    for row in rows:
        if (row['status'] != 'ok' or not row['converged'] or not row['loss_matches_history_min']
                or row.get('initial_baseline_status') != 'ok'
                or not row.get('initial_converged')):
            raise ValueError(f"Unvalidated trained/initial result: {row['id']}")
        stem = row['checkpoint_stem']
        record_path = args.models_root/'training_records'/(stem+'.json')
        record = json.loads(record_path.read_text())
        if record['initialization']['seed'] != 42 or record['basinhopping']['niter'] != 4:
            raise ValueError('Training configuration does not match this campaign')
        history = np.array([h['train_loss'] for h in record['history']])
        np.testing.assert_allclose(history.min(), row['checkpoint_loss'], rtol=1e-5, atol=1e-6)
        np.testing.assert_equal(record['selection']['train_loss'], history.min())
        ss = np.random.SeedSequence(42).spawn(2)[0]
        initial = np.random.default_rng(ss).random(record['n_parameters'])
        np.testing.assert_array_equal(initial, record['initialization']['parameters'])
        stored = args.models_root/'pickle_json'/(stem+'.json')
        model = json.loads(stored.read_text())
        errors = {}
        for field, location in (
            ('tunable_parameters', args.models_root/'params'/(stem+'.npy')),
            ('S_l', args.models_root/'trained_operators'/(stem+'_S_l.npy')),
            ('D_l', args.models_root/'trained_operators'/(stem+'_D_l.npy'))):
            a, b = _serialized_array(model[field]), load_numeric(location)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
            errors[field] = float(np.max(abs(a-b)))
        if model['training_record']['selection'] != record['selection']:
            raise ValueError('Serialized model and training record selection disagree')
        tested = args.models_root/'testing_results'/(stem+'.txt')
        moments = np.array(row['points'][0]['moments_real'])+1j*np.array(row['points'][0]['moments_imag'])
        archived_moments = load_numeric(tested).reshape(-1)
        np.testing.assert_allclose(moments, archived_moments, rtol=1e-7, atol=1e-7)
        checks.append(dict(id=row['id'], serialized_array_max_errors=errors,
                           saved_vs_fock_moment_max_error=float(max(abs(moments-archived_moments))),
                           initialization_regenerated_exactly=True,
                           seed=42, basinhopping_proposals=4,
                           local_minimization_failures=record['scipy_result']['minimization_failures'],
                           selected_event=record['selection'],
                           serialized_model_sha256=hashlib.sha256(stored.read_bytes()).hexdigest(),
                           testing_moments_sha256=hashlib.sha256(tested.read_bytes()).hexdigest()))
    _atomic_text(args.evaluation/'checkpoint_integrity.json', json.dumps(checks,indent=2)+'\n')
    _atomic_text(args.evaluation/'cat_table_manuscript.tex', make_table(rows, 'tab:catstate-results'))
    _atomic_text(args.evaluation/'cat_table_all_configurations.tex', make_table(rows, 'tab:catstate-results-all'))
    diagnostics = []
    for row, check in zip(rows, checks):
        point = row['points'][0]
        diagnostics.append(dict(id=row['id'], N=row['N'], L=row['L'],
            subtractions_zero_based=row['subtractions'], training_loss=row['loss_history_min'],
            recomputed_loss=row['checkpoint_loss'], initial_loss=row['initial_checkpoint_loss'],
            initial_fidelity=row['initial_fidelity_mean'], fidelity=row['fidelity_mean'],
            cutoff=row['cutoff'], initial_cutoff=row['initial_cutoff'],
            purity=point['purity'], parity=point['parity'],
            even_photon_probability=point['even_photon_probability'],
            mean_photon_number=point['mean_photon_number'],
            ideal_operator_weight=point['ideal_operator_weight'],
            local_minimization_failures=check['local_minimization_failures']))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
    writer.writeheader(); writer.writerows(diagnostics)
    _atomic_text(args.evaluation/'cat_diagnostics.csv', stream.getvalue())
    print('Eight checkpoints validated; paper tables and diagnostics exported.')


if __name__ == '__main__':
    main()
