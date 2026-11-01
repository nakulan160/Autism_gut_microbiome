#!/usr/bin/env python3
"""Regenerate the Waypoint/AGORA2 input NPZ from the audited corrected species TSV.

Preserves the taxonomy, AGORA2 annotations, Waypoint vocabulary, labels, folds and
sample order from the reference NPZ. Recomputes EVERY abundance-dependent field.
No embeddings, training or metabolic simulations are performed.

Usage:
    python rebuild_waypoint_agora_npz.py
or:
    python rebuild_waypoint_agora_npz.py --tsv corrected.tsv --reference old.npz --out new.npz

Requires only numpy and Python's standard library.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, delimiter='\t')
        header = next(reader)
        rows = list(reader)
    if not header or header[0] != 'Run ID':
        raise ValueError('TSV first column must be Run ID')
    if any(len(row) != len(header) for row in rows):
        raise ValueError('TSV has a malformed row')
    ids = [row[0] for row in rows]
    X = np.array([[float(value) for value in row[1:]] for row in rows], dtype=np.float64)
    if not np.isfinite(X).all() or (X < 0).any():
        raise ValueError('Non-finite or negative abundance')
    if len(set(ids)) != len(ids) or len(set(header[1:])) != len(header)-1:
        raise ValueError('Duplicate run IDs or feature columns')
    return ids, header[1:], X


def make_derived(X, z):
    """Rebuild abundance-derived arrays with the original stored taxonomy rules."""
    n, p = X.shape
    result = {
        'original_X_percent': X,
        'original_X_fraction': X / 100.0,
    }
    orig_names = z['original_feature_names']
    unk = np.flatnonzero(orig_names == 'Unknown')
    if len(unk) != 1:
        raise ValueError('Expected one Unknown abundance column')

    total = X.sum(axis=1)
    unknown = X[:, int(unk[0])].copy()
    named = total - unknown
    if np.any(named <= 0) or np.any(total <= 0):
        raise ValueError('Encountered a run with zero total/named mass')
    result.update({
        'Unknown_percent': unknown,
        'named_mass_percent': named,
        'original_total_mass_percent': total,
    })

    sp_mask = z['input_species_agora_matched'].astype(bool)
    gn_mask = z['input_genus_agora_matched'].astype(bool)
    sp_matched = X[:, sp_mask].sum(axis=1)
    gn_matched = X[:, gn_mask].sum(axis=1)

    # Resolved means the stored taxonomic lineage has a species segment,
    # NOT simply current_ranks == 'species'. The latter is not equivalent.
    orig_paths = z['original_feature_taxonomy_paths']
    resolved = np.asarray([any(segment.startswith('s__') for segment in str(p).split(';'))
                           for p in orig_paths], dtype=bool)
    sp_mass = X[:, resolved].sum(axis=1)
    result.update({
        'agora_species_matched_percent': sp_matched,
        'agora_species_unmatched_named_percent': named-sp_matched,
        'agora_species_coverage_fraction_named': sp_matched/named,
        'agora_species_coverage_fraction_total': sp_matched/total,
        'species_resolved_mass_percent': sp_mass,
        'species_resolution_available': sp_mass > 0,
        'agora_species_coverage_fraction_species': np.divide(
            sp_matched, sp_mass,
            out=np.full(n, np.nan), where=sp_mass > 0),
        'agora_species_unmatched_resolved_percent': sp_mass-sp_matched,
        'agora_genus_matched_percent': gn_matched,
        'agora_genus_unmatched_named_percent': named-gn_matched,
        'agora_genus_coverage_fraction_named': gn_matched/named,
        'agora_genus_coverage_fraction_total': gn_matched/total,
    })

    # Remove species segments, then group into the *existing* fixed Waypoint
    # rank/genus vocabulary. Four unresolved named features remain unresolved.
    projection = [';'.join(s for s in str(p).split(';') if s and not s.startswith('s__'))
                  for p in orig_paths]
    candidate_paths = z['waypoint_taxonomy_paths']
    path_index = {str(path): i for i, path in enumerate(candidate_paths)}
    if len(path_index) != len(candidate_paths):
        raise ValueError('Reference Waypoint paths are not unique')
    wp_raw = np.zeros((n, len(candidate_paths)), dtype=np.float64)
    unres = np.zeros(n, dtype=np.float64)
    for feature_index, path in enumerate(projection):
        if str(orig_names[feature_index]) == 'Unknown':
            continue
        if path in path_index:
            wp_raw[:, path_index[path]] += X[:, feature_index]
        else:
            unres += X[:, feature_index]

    usable = wp_raw.sum(axis=1)
    if np.any(usable <= 0):
        raise ValueError('Waypoint usable mass is zero for at least one run')
    wp_relative = wp_raw / usable[:, None]
    wp_fraction_named = wp_raw / named[:, None]

    token_ids = z['waypoint_candidate_token_ids']
    order = np.argsort(token_ids, kind='stable')
    known_indices = order[token_ids[order] > 1]
    if not np.array_equal(token_ids[known_indices], z['waypoint_recognized_token_ids']):
        raise ValueError('Recognized token order differs from the frozen reference')
    recognized_raw = wp_raw[:, known_indices]
    recognized_mass = recognized_raw.sum(axis=1)

    result.update({
        'waypoint_taxonomy_unresolved_percent': unres,
        'waypoint_usable_mass_percent': usable,
        'waypoint_X_relative': wp_relative,
        'waypoint_X_fraction_named': wp_fraction_named,
        'waypoint_recognized_X_relative': wp_relative[:, known_indices],
        'waypoint_recognized_X_fraction_named': wp_fraction_named[:, known_indices],
        'waypoint_coverage_fraction_named': recognized_mass/named,
        'waypoint_coverage_fraction_total': recognized_mass/total,
        'waypoint_vocab_unmatched_percent': named-recognized_mass,
        'waypoint_vocab_oov_usable_percent': usable-recognized_mass,
    })
    # Strict independent checks on group sums and tokenizer coverage.
    np.testing.assert_allclose(usable + unres, named, atol=1e-9, rtol=0)
    np.testing.assert_allclose(wp_relative.sum(axis=1), 1, atol=1e-12, rtol=0)
    return result


def verify_reconstruction(z):
    """Prove we reproduce the archived inputs before applying the correction."""
    original = make_derived(z['original_X_percent'], z)
    for k, proposed in original.items():
        current = z[k]
        if proposed.dtype == bool:
            np.testing.assert_array_equal(proposed, current, err_msg=k)
        else:
            np.testing.assert_allclose(proposed, current, atol=1e-10, rtol=0,
                                       equal_nan=True, err_msg=k)
    return len(original)


def verify_audit(X, z):
    """Reject any corrected TSV differing from the 46 archived candidate edits."""
    review = json.loads(z['source_duplicate_review_json'].item())
    changes = review['changes']
    if len(changes) != 46 or len({e['run_id'] for e in changes}) != 45:
        raise ValueError('Incorrect expected number of source-audit groups')
    original = z['original_X_percent']
    expected = original.copy()
    ids = {str(s): i for i, s in enumerate(z['sample_ids'])}
    features = {str(s): i for i, s in enumerate(z['original_feature_names'])}
    taxids = z['current_taxids']
    authorized = np.zeros(X.shape, dtype=bool)
    for e in changes:
        i = ids[e['run_id']]
        js = [features[s] for s in e['original_features']]
        if not all(int(taxids[j]) == int(e['taxid']) for j in js):
            raise ValueError('Source audit TaxID mismatch')
        np.testing.assert_allclose(original[i, js].sum(), e['original_group_sum_percent'], atol=1e-10, rtol=0)
        keep = float(e['candidate_deduplicated_percent'])
        np.testing.assert_allclose(original[i, js].sum(), 2*keep, atol=1e-10, rtol=0)
        expected[i, js[0]] = keep
        for j in js[1:]:
            expected[i, j] = 0
        authorized[i, js] = True
    np.testing.assert_array_equal(X, expected, err_msg='Corrected TSV is not the audit-approved candidate matrix')
    np.testing.assert_array_equal(X[~authorized], original[~authorized])
    return len(changes), len({e['run_id'] for e in changes}), int(np.count_nonzero(X != original))


def main():
    directory = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tsv', type=Path, default=directory/'species_model_input_matrix_corrected.tsv')
    p.add_argument('--reference', type=Path, default=directory/'waypoint_agora_checked(1).npz')
    p.add_argument('--out', type=Path, default=directory/'waypoint_agora_corrected.npz')
    args = p.parse_args()
    if args.out.resolve() in (args.tsv.resolve(), args.reference.resolve()):
        raise ValueError('Refusing to overwrite an input')
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {args.out}')

    runs, features, X = read_tsv(args.tsv)
    with np.load(args.reference, allow_pickle=False) as z:
        if runs != z['sample_ids'].tolist() or features != z['original_feature_names'].tolist():
            raise ValueError('Corrected TSV does not match the reference sample/feature order')
        if X.shape != (389, 1068):
            raise ValueError(f'Unexpected matrix size: {X.shape}')
        reproduced = verify_reconstruction(z)
        n_groups, n_runs, n_cells = verify_audit(X, z)
        derived = make_derived(X, z)
        original_settings = json.loads(z['settings_json'].item())
        new_settings = dict(original_settings)
        new_settings.update({
            'schema': 'asd_waypoint_agora_inputs_v2_corrected',
            'abundance_review': 'archived 46-group candidate duplicate correction applied; original source-row independent verification pending',
            'abundance_training_ready': False,
            'abundance_training_ready_reason': 'primary source records have not been independently re-audited; suitable for documented exploratory analysis',
            'source_duplicate_correction_applied': True,
            'corrected_tsv_name': args.tsv.name,
            'corrected_tsv_sha256': sha256(args.tsv),
            'source_duplicate_changed_cells': n_cells,
            'original_npz_sha256': sha256(args.reference),
            'waypoint_empty_run_count': int(np.sum(derived['waypoint_recognized_X_relative'].sum(axis=1) == 0)),
        })
        review = json.loads(z['source_duplicate_review_json'].item())
        review['previous_status'] = review['status']
        review['status'] = 'archived candidate corrections applied to current matrix; raw source-record audit pending'
        review['applied_to_current_matrix'] = True
        review['corrected_tsv_sha256'] = sha256(args.tsv)
        review['changed_cells'] = n_cells
        source_settings = json.loads(z['source_settings_json'].item())
        source_settings['corrected_matrix_file'] = args.tsv.name
        source_settings['corrected_matrix_sha256'] = sha256(args.tsv)
        source_settings['matrix_source_files_are_historical'] = True

        output = {k: z[k] for k in z.files}
        output.update(derived)
        output['settings_json'] = np.array(json.dumps(new_settings, sort_keys=True))
        output['source_duplicate_review_json'] = np.array(json.dumps(review, sort_keys=True))
        output['source_settings_json'] = np.array(json.dumps(source_settings, sort_keys=True))

    totals = derived['original_total_mass_percent']
    if (np.abs(totals - 100) > .0002).any():
        raise ValueError('Post-correction row sums do not close near 100%')
    if not np.array_equal(output['source_duplicate_run_flag'],
                          np.load(args.reference, allow_pickle=False)['source_duplicate_run_flag']):
        raise ValueError('Historical duplicate flags changed')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **output)
    with np.load(args.out, allow_pickle=False) as check:
        if set(check.files) != set(output):
            raise ValueError('Keys lost during archive writing')
        for k in derived:
            np.testing.assert_array_equal(check[k], derived[k], err_msg=k)
        assert json.loads(check['settings_json'].item())['source_duplicate_correction_applied'] is True
        assert np.array_equal(check['labels'], output['labels'])
        assert np.array_equal(check['random_outer_fold'], output['random_outer_fold'])
        assert np.array_equal(check['waypoint_candidate_token_ids'], output['waypoint_candidate_token_ids'])
        print('Reconstructed reference arrays accurately:', reproduced)
        print('Corrected groups/runs/numerical cells:', n_groups, n_runs, n_cells)
        print('Matrix:', check['original_X_percent'].shape, '| NPZ entries:', len(check.files))
        print('Labels: ASD', int((check['labels'] == 1).sum()), '| Healthy', int((check['labels'] == 0).sum()))
        print('Totals range: %.7f%%–%.7f%%' % (totals.min(), totals.max()))
        print('Waypoint recognized shape:', check['waypoint_recognized_X_relative'].shape)
        print('Waypoint named-mass coverage median: %.4f%%' % (100*np.median(check['waypoint_coverage_fraction_named'])))
        print('AGORA genus coverage median: %.4f%%' % (100*np.median(check['agora_genus_coverage_fraction_named'])))
        print('AGORA species coverage median: %.4f%%' % (100*np.median(check['agora_species_coverage_fraction_named'])))
        print('Saved:', args.out)
        print('SHA-256:', sha256(args.out))


if __name__ == '__main__':
    main()
