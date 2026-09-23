"""Read-only terminal residual units/conditioning audit; no QP or fit execution."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import capture_failed_qp as cap

ROOT = Path(__file__).resolve().parent
INPUTS = [
    ('capture-below-a/imported/results/capture-below_one-64.json',
     '4958d20fdaf3f7d047ccd14a75dfb4a5ffadebd2f94c1ee8514a33fecda1b0af'),
    ('capture-mixed-b/imported/results/capture-mixed_support-256.json',
     'd457d0be06291349cefcce142aff8b8beeeb4f1cb88ab77e4da0acd69347cd26'),
]


def load(receipt, descriptor):
    return cap.load_tensor(receipt, descriptor, 'cpu').numpy()


def main():
    rows = []
    for relative, digest in INPUTS:
        receipt = cap.load_capture(ROOT / relative, digest)
        for entry in receipt['captures']:
            record = cap.load_record(receipt, entry)
            h = load(receipt, record['problem']['h'])
            h_reference = float(np.sort(h)[(len(h) - 1) // 2])  # torch.median convention
            state = {name: load(receipt, record['terminal_admm'][name]) for name in ('x', 'z', 'v', 'q', 'rho')}
            x, z, rho = state['x'], state['z'], float(state['rho'])
            differences = x[None, :] - x[:, None]
            primal = float(np.sqrt(.5 * np.square(differences - z).sum()))
            n = len(x)
            edges = n * (n - 1) // 2
            context = {key: record['context'][key] for key in ('path_index', 'start_index', 'lambda_value', 'inflation', 'backtrack_index')}
            rows.append(dict(fixture=receipt['fixture_family'], nodes=n, capture_receipt=relative,
                capture_receipt_sha256=digest, record=entry['record'], record_sha256=entry['sha256'],
                context=context, h_min=float(h.min()), h_max=float(h.max()),
                h_median=h_reference, terminal_rho=rho, terminal_rho_over_h_median=rho/h_reference,
                node_consensus_curvature_ratio=n*rho/h_reference,
                terminal_primal_edge_l2=primal, terminal_primal_edge_rms=primal/np.sqrt(edges),
                terminal_dual_multiplier=rho,
                objective_invariant_dual_multiplier=rho/h_reference,
                edge_node_rms_invariant_dual_multiplier=rho/h_reference/np.sqrt(n),
                actual_dual_norm=None,
                actual_dual_norm_missing_reason='previous_z was not part of the exact terminal capture; no final dual residual or balancing decision is reconstructed.',
                terminal_q_rho_v_max_difference=float(np.abs(state['q']-rho*state['v']).max()),
                state_scope='rho/v can include residual balancing after the last clipped q; q=rho*v is not assumed.',
            ))
    result = dict(schema='clipp1d.cuda.residual_balance_array_audit.v1',
        scope='CPU array arithmetic only on13 hash-bound failed CUDA QPs; no fit/recovery, controller trajectory, speed or CUDA qualification.',
        script_sha256=cap.common.sha(Path(__file__)), capture_loader_sha256=cap.common.sha(Path(cap.__file__)),
        objective_scaling_proof=dict(
            transformation='For c>0: h,caps,q,rho -> c times their values; target,boxes,x,z,v unchanged. Multiply the entire QP objective by c.',
            admm_primal='h*target + rho*D^T(z-v) and h+rho*D^T D both scale by c, so the exact boxed x update is unchanged.',
            admm_edges='caps/rho and scaled v are unchanged, so the z/v updates are unchanged.',
            current_controller='r=||Dx-z||_edge is unchanged; s=rho||D^T(z-zprev)||_node scales by c. The existing factor10 comparison is not invariant.',
            fixed_curvature_controller='Use r versus s/h_ref with h_ref=lower-median(h)>0 fixed for the QP. Both quantities have CCF units and are unchanged under c.',
            optional_dimension_normalization='Compare r/sqrt(M) versus s/(h_ref*sqrt(N)), M=N(N-1)/2. These are edge and node RMS quantities; objective-scale invariance is preserved.',
            conditioning_limits='Neither normalization proves fastest convergence or general unit/coordinate invariance. h_ref does not remove heterogeneous node curvature or graph-degree effects.',
            state_update='Retain rho bounds relative to its initial value and v<-v*(oldrho/newrho); physical dual initialization and all original certificates remain separate.',
            gates='Original absolute-plus-relative gap and KKT gates must be unchanged. Their absolute gap term is intentionally not scale invariant; controller invariance is not permission to rescale scientific admission.',
        ), rows=rows)
    out = ROOT / 'residual-balance-diagnosis.json'
    cap.common.write_json(out, result)
    print(json.dumps(dict(output=str(out), sha256=cap.common.sha(out), rows=len(rows))))
    for row in rows:
        print(row['fixture'], row['context']['path_index'], row['context']['start_index'],
              'hmed',row['h_median'],'rho',row['terminal_rho'],
              'rho/hmed',row['terminal_rho_over_h_median'],
              'N*rho/hmed',row['node_consensus_curvature_ratio'],
              'primal',row['terminal_primal_edge_l2'])


if __name__ == '__main__':
    main()
