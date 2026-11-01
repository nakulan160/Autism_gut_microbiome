#!/usr/bin/env python3
"""Step 4: strain-level, diet-conditional secretion tests for 27 matched taxa.

Inputs beside script: AGORA2_SBML.zip, WesternDietAGORA2.txt,
AGORA2_Baseline_XGBoost_Top50_species_model_ids.tsv
Output (ONLY ONE): AGORA2_27_species_analysis/04_strain_secretion.tsv
Results describe model feasibility, NOT measured production or a validated IMPC.
"""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import io
import os
import subprocess

import cobra
from cobra.util.solver import linear_reaction_coefficients

BASE = Path(__file__).resolve().parent
ZIP = BASE / 'AGORA2_SBML.zip'
DIET_FILE = BASE / 'WesternDietAGORA2.txt'
MAPPING = BASE / 'AGORA2_Baseline_XGBoost_Top50_species_model_ids.tsv'
OUT = BASE / 'AGORA2_27_species_analysis' / '04_strain_secretion.tsv'

# Predefined gut-microbial products / gut-brain-axis candidate compounds.
# These names and reaction IDs were checked against the Step 2 mapping.
TARGETS = {
    'Acetate': 'EX_ac(e)',
    'Propionate': 'EX_ppa(e)',
    'Butyrate': 'EX_but(e)',
    'Succinate': 'EX_succ(e)',
    'GABA': 'EX_4abut(e)',
    'Indole': 'EX_indole(e)',
    'p-Cresol': 'EX_pcresol(e)',
    'Indole-3-propionate': 'EX_ind3ppa(e)',
    'Indole-3-acetate': 'EX_ind3ac(e)',
    'Phenylacetate': 'EX_pac(e)',
    'D-lactate': 'EX_lac_D(e)',
    'L-lactate': 'EX_lac_L(e)',
}
GROWTH_FRACTION = 0.10   # sensitivity assumption, NOT measured growth
TOL = 1e-7
WORKERS = int(os.environ.get('AGORA_WORKERS', '4'))
SMOKE_TEST = os.environ.get('AGORA_SMOKE', '0') == '1'

FIELDS = [
    'Baseline_Rank', 'Input_Taxon', 'AGORA2_MicrobeID', 'Metabolite',
    'Exchange_ID', 'Diet_SHA256', 'Diet_reactions_matched', 'Repaired_invalid_bytes',
    'No_diet_growth_control', 'Growth_on_diet', 'Growth_with_target_import_blocked',
    'Required_growth', 'No_diet_max_secretion', 'Max_secretion',
    'Native_source_reactions', 'Active_sources_at_secretion_optimum',
    'Status', 'Error'
]


def read_diet():
    diet = {}
    for line_no, raw in enumerate(DIET_FILE.read_text(encoding='utf-8-sig').splitlines(), 1):
        a = raw.strip().split()
        if not a or raw.lstrip().startswith(('#', '%')):
            continue
        if len(a) != 2 or not a[0].startswith('EX_'):
            raise ValueError(f'Bad diet entry at line {line_no}: {raw!r}')
        bound = float(a[1])
        if not (-1000 <= bound <= 0) or a[0] in diet:
            raise ValueError(f'Invalid or repeated diet entry: {raw!r}')
        diet[a[0]] = bound
    if len(diet) < 10:
        raise ValueError('Too few entries in dietary medium')
    return diet


def load_model(mid):
    p = subprocess.run(['unzip', '-p', str(ZIP), mid + '.xml'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode or not p.stdout:
        raise ValueError(f'ZIP extraction failed: {p.stderr.decode(errors="replace")[-180:]}')
    txt = p.stdout.decode('utf-8', errors='surrogateescape')
    invalid = Counter(ord(c) - 0xDC00 for c in txt if 0xDC80 <= ord(c) <= 0xDCFF)
    if invalid:
        replacement = {0xDC00 + b: bytes([b]).decode('cp1252') for b in invalid}
        txt = ''.join(replacement.get(ord(c), c) for c in txt)
    return cobra.io.read_sbml_model(io.StringIO(txt)), sum(invalid.values())


def coefficient(r):
    if len(r.metabolites) != 1:
        raise ValueError(f'Unexpected boundary stoichiometry: {r.id}')
    c = next(iter(r.metabolites.values()))
    if abs(c) < TOL:
        raise ValueError(f'Zero boundary coefficient: {r.id}')
    return c


def close_import(r):
    # Prevent the reaction from adding material into the network.
    if coefficient(r) < 0:
        if r.upper_bound < -TOL:
            raise ValueError(f'Forced import in {r.id}')
        r.lower_bound = max(r.lower_bound, 0)
    else:
        if r.lower_bound > TOL:
            raise ValueError(f'Forced internal source in {r.id}')
        r.upper_bound = min(r.upper_bound, 0)


def apply_diet(model, diet):
    exchanges = {r.id: r for r in model.exchanges}
    for r in exchanges.values():
        close_import(r)
    matched = 0
    for rid, lower in diet.items():
        r = exchanges.get(rid)
        if r is None:
            continue
        if coefficient(r) >= 0 or next(iter(r.metabolites)).compartment != 'e':
            raise ValueError(f'Unexpected diet exchange orientation: {rid}')
        r.lower_bound = lower
        matched += 1
    if 'EX_o2(e)' in exchanges:
        close_import(exchanges['EX_o2(e)'])
    if matched < 10:
        raise ValueError(f'Only {matched} diet reactions match model')
    return matched


def biomass_reaction(model):
    coeffs = list(linear_reaction_coefficients(model).items())
    if len(coeffs) != 1 or coeffs[0][1] <= 0:
        raise ValueError('Biomass objective is not one positive reaction')
    return coeffs[0][0]


def max_flux(model, reaction):
    model.objective = reaction
    model.objective_direction = 'max'
    result = model.optimize()
    if result.status != 'optimal':
        return result.status, None, None
    return result.status, float(result.fluxes[reaction.id]), result


def growth(model, biomass):
    status, value, _ = max_flux(model, biomass)
    return status, value


def internal_sources(model):
    exchange_ids = {r.id for r in model.exchanges}
    result = []
    for r in model.boundary:
        if r.id in exchange_ids:
            continue
        c = coefficient(r)
        if (c > 0 and r.upper_bound > TOL) or (c < 0 and r.lower_bound < -TOL):
            result.append(r)
    return result


def value(x):
    return '' if x is None else f'{x:.9g}'


def evaluate_one(task, diet, diet_sha):
    rank, species, mid = task
    common = {'Baseline_Rank': rank, 'Input_Taxon': species,
              'AGORA2_MicrobeID': mid, 'Diet_SHA256': diet_sha}
    rows = []
    try:
        model, repaired = load_model(mid)
        biomass = biomass_reaction(model)
        matched = apply_diet(model, diet)
        sources = internal_sources(model)
        source_names = ';'.join(r.id for r in sources)
        with model:
            for r in model.exchanges:
                close_import(r)
            blank_status, blank_growth = growth(model, biomass)
        if blank_status != 'optimal':
            blank_control = 'REVIEW_' + blank_status
        elif blank_growth is not None and blank_growth <= TOL:
            blank_control = 'PASS_zero_growth'
        else:
            blank_control = 'FAIL_growth_without_diet'
        main_status, main_growth = growth(model, biomass)
        exchange_map = {r.id: r for r in model.exchanges}

        for name, rid in TARGETS.items():
            row = dict(common, Metabolite=name, Exchange_ID=rid,
                       Diet_reactions_matched=matched, Repaired_invalid_bytes=repaired,
                       No_diet_growth_control=blank_control, Growth_on_diet=value(main_growth),
                       Growth_with_target_import_blocked='', Required_growth='',
                       No_diet_max_secretion='', Max_secretion='',
                       Native_source_reactions=source_names,
                       Active_sources_at_secretion_optimum='', Status='', Error='')
            if rid not in exchange_map:
                row['Status'] = 'EXCHANGE_ABSENT'
            elif coefficient(exchange_map[rid]) >= 0:
                row['Status'] = 'REVIEW_exchange_orientation'
            elif blank_control != 'PASS_zero_growth':
                row['Status'] = 'REVIEW_failed_no_diet_growth_control'
            elif main_status != 'optimal' or main_growth is None or main_growth <= TOL:
                row['Status'] = 'NOT_EVALUABLE_no_growth_on_diet'
            else:
                r = exchange_map[rid]
                with model:
                    for ex in model.exchanges:
                        close_import(ex)
                    b_status, b_flux, _ = max_flux(model, r)
                row['No_diet_max_secretion'] = value(b_flux)
                if b_status != 'optimal' or b_flux is None or b_flux > TOL:
                    row['Status'] = 'REVIEW_secretion_without_diet'
                else:
                    with model:
                        close_import(r)
                        g_status, g_value = growth(model, biomass)
                        row['Growth_with_target_import_blocked'] = value(g_value)
                        if g_status != 'optimal' or g_value is None or g_value <= TOL:
                            row['Status'] = 'NOT_EVALUABLE_no_growth_after_target_block'
                        else:
                            required = GROWTH_FRACTION * g_value
                            row['Required_growth'] = value(required)
                            biomass.lower_bound = max(biomass.lower_bound, required)
                            s_status, f, solution = max_flux(model, r)
                            row['Max_secretion'] = value(f)
                            if s_status != 'optimal' or f is None:
                                row['Status'] = 'REVIEW_secretion_' + s_status
                            elif f > TOL and f >= r.upper_bound - TOL:
                                row['Status'] = 'REVIEW_exchange_bound_reached'
                            elif f > TOL:
                                row['Status'] = 'MODEL_FEASIBLE_conditional_native_sources'
                                row['Active_sources_at_secretion_optimum'] = ';'.join(
                                    f'{src.id}:{solution.fluxes[src.id]:.5g}' for src in sources
                                    if abs(float(solution.fluxes[src.id])) > TOL)
                            else:
                                row['Status'] = 'NO_SECRETION_under_tested_conditions'
            rows.append(row)

    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'[:250]
        rows = [dict(common, Metabolite=name, Exchange_ID=rid,
                     Diet_reactions_matched='', Repaired_invalid_bytes='',
                     No_diet_growth_control='', Growth_on_diet='',
                     Growth_with_target_import_blocked='', Required_growth='',
                     No_diet_max_secretion='', Max_secretion='',
                     Native_source_reactions='', Active_sources_at_secretion_optimum='',
                     Status='MODEL_ERROR', Error=error)
                for name, rid in TARGETS.items()]
    return mid, rows


def main():
    for p in (ZIP, DIET_FILE, MAPPING):
        if not p.is_file():
            raise SystemExit(f'Missing required file: {p}')
    if WORKERS < 1 or WORKERS > 10:
        raise SystemExit('AGORA_WORKERS must be between 1 and 10')
    diet = read_diet()
    digest = hashlib.sha256(DIET_FILE.read_bytes()).hexdigest()
    with MAPPING.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter='\t')
        tasks = [(r['Baseline_Rank'], r['Input_Taxon'], r['AGORA2_MicrobeID']) for r in reader]
    if len(tasks) != 1473 or len(set(t[2] for t in tasks)) != 1473 or len(set(t[1] for t in tasks)) != 27:
        raise SystemExit('Unexpected species/model count: please check mapping')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if OUT.exists():
        with OUT.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            if reader.fieldnames != FIELDS:
                raise SystemExit('Existing output has incompatible columns')
            counts = Counter()
            for row in reader:
                if row['Diet_SHA256'] != digest:
                    raise SystemExit('Existing output uses a different diet; do not mix runs')
                counts[row['AGORA2_MicrobeID']] += 1
            if any(n != len(TARGETS) for n in counts.values()):
                raise SystemExit('Incomplete model rows in output; inspect before resuming')
            completed = set(counts)
    if SMOKE_TEST:
        # First listed strain from each species: a technical test, NOT representative strain selection.
        seen_species = set()
        trial = []
        for task in tasks:
            if task[1] not in seen_species:
                trial.append(task)
                seen_species.add(task[1])
        remaining = [task for task in trial if task[2] not in completed]
    else:
        remaining = [task for task in tasks if task[2] not in completed]
        print('SMOKE TEST ONLY: one available model per species; do not interpret as a species consensus.')
    print(f'27 species; 1473 models; {len(TARGETS)} predefined metabolites')
    print(f'Western diet: {len(diet)} entries; SHA256={digest}')
    print(f'Workers: {WORKERS}; already completed: {len(completed)}; remaining: {len(remaining)}')
    print('Only model-feasible secretion is evaluated; internal pseudo-sources are retained and audited.')
    if not remaining:
        print('Nothing to rerun; output is complete:', OUT)
        return
    with OUT.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter='\t')
        if f.tell() == 0:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(evaluate_one, task, diet, digest) for task in remaining]
            errors = 0
            for i, future in enumerate(as_completed(futures), 1):
                mid, rows = future.result()
                writer.writerows(rows)
                f.flush()
                if rows[0]['Status'] == 'MODEL_ERROR':
                    errors += 1
                    print('REVIEW MODEL_ERROR:', mid, rows[0]['Error'], flush=True)
                if i % 25 == 0 or i == len(remaining):
                    print(f'Processed this run: {i}/{len(remaining)}; model errors: {errors}', flush=True)
    print('Completed. One TSV:', OUT)
    print('Review MODEL_ERROR, REVIEW_* and pseudo-source dependencies before computing IMPC.')


if __name__ == '__main__':
    main()
