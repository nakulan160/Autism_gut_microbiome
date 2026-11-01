#!/usr/bin/env python3
"""AGORA2 Step 3: native-reconstruction secretion feasibility and source audit.

Keeps original non-exchange reactions and applies the published AGORA2 Western diet
only to extracellular EX_ reactions. Source blocking is a sensitivity diagnostic,
NOT an asserted biological correction. No results are metabolite measurements.

Inputs beside script: AGORA2_SBML.zip, WesternDietAGORA2.txt
One output: AGORA2_27_species_analysis/03_alistipes_secretion_pilot.tsv
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import io
import subprocess

import cobra
from cobra.util.solver import linear_reaction_coefficients

BASE = Path(__file__).resolve().parent
ARCHIVE = BASE / "AGORA2_SBML.zip"
DIET_PATH = BASE / "WesternDietAGORA2.txt"
OUTPUT = BASE / "AGORA2_27_species_analysis" / "03_alistipes_secretion_pilot.tsv"
STRAINS = (
    "Alistipes_indistinctus_ERR2221377",
    "Alistipes_indistinctus_YIT_12060",
)
TARGETS = {
    "Acetate": "EX_ac(e)",
    "Succinate": "EX_succ(e)",
    "Indole": "EX_indole(e)",
    "GABA": "EX_4abut(e)",
    "Propionate": "EX_ppa(e)",
    "Butyrate": "EX_but(e)",
}
TOL = 1e-7
GROWTH_FRACTION = 0.10    # explicitly an assumption, not measured growth


def read_diet(path):
    diet = {}
    for i, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = raw.strip().split()
        if not parts or raw.lstrip().startswith(("#", "%")):
            continue
        if len(parts) != 2 or not parts[0].startswith("EX_"):
            raise ValueError(f"Unexpected diet entry at line {i}: {raw!r}")
        name, value = parts
        lower = float(value)
        if not -1000 <= lower <= 0 or name in diet:
            raise ValueError(f"Invalid or duplicate diet entry at line {i}: {raw!r}")
        diet[name] = lower
    if len(diet) < 10:
        raise ValueError("Diet file contains too few reactions")
    return diet


def read_model(mid):
    proc = subprocess.run(
        ["unzip", "-p", str(ARCHIVE), mid + ".xml"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise RuntimeError(f"unzip failed for {mid}: {proc.stderr.decode(errors='replace')[-400:]}")
    text = proc.stdout.decode("utf-8", errors="surrogateescape")
    invalid = Counter(ord(c)-0xDC00 for c in text if 0xDC80 <= ord(c) <= 0xDCFF)
    for b in invalid:
        bytes((b,)).decode("cp1252")  # fail if character mapping is ambiguous
    if invalid:
        text = "".join(
            bytes((ord(c)-0xDC00,)).decode("cp1252")
            if 0xDC80 <= ord(c) <= 0xDCFF else c
            for c in text
        )
    model = cobra.io.read_sbml_model(io.StringIO(text))
    return model, sum(invalid.values())


def boundary_coeff(reaction):
    if len(reaction.metabolites) != 1:
        raise ValueError(f"Unexpected multi-metabolite boundary: {reaction.id}")
    coeff = next(iter(reaction.metabolites.values()))
    if abs(coeff) < 1e-12:
        raise ValueError(f"Zero stoichiometry: {reaction.id}")
    return coeff


def block_import(reaction):
    """Close the direction in which a one-metabolite boundary ADDS mass."""
    s = boundary_coeff(reaction)
    if s < 0:
        if reaction.upper_bound < 0:
            raise ValueError(f"Forced import in {reaction.id}: {reaction.bounds}")
        reaction.lower_bound = max(0, reaction.lower_bound)
    else:
        if reaction.lower_bound > 0:
            raise ValueError(f"Forced mass source in {reaction.id}: {reaction.bounds}")
        reaction.upper_bound = min(0, reaction.upper_bound)


def set_diet(model, diet):
    """Apply dietary limits ONLY to extracellular nutrient exchange reactions."""
    exchanges = {r.id: r for r in model.exchanges}
    for rxn in exchanges.values():
        block_import(rxn)
    matched = 0
    for rid, lower in diet.items():
        if rid not in exchanges:
            continue
        rxn = exchanges[rid]
        if (boundary_coeff(rxn) >= 0 or
                next(iter(rxn.metabolites)).compartment != "e"):
            raise ValueError(f"Diet reaction has unexpected convention: {rid}")
        rxn.lower_bound = lower
        matched += 1
    if "EX_o2(e)" in exchanges:
        block_import(exchanges["EX_o2(e)"])
    if matched < 10:
        raise ValueError(f"Only {matched}/{len(diet)} diet reactions match model")
    return matched


def nonexchange_sources(model):
    """Return single-metabolite non-exchange boundaries that can CREATE mass."""
    exchange_ids = {x.id for x in model.exchanges}
    found = []
    for r in model.boundary:
        if r.id in exchange_ids:
            continue
        s = boundary_coeff(r)
        if (s > 0 and r.upper_bound > TOL) or (s < 0 and r.lower_bound < -TOL):
            found.append(r)
    return found


def close_nonexchange_sources(model):
    sources = nonexchange_sources(model)
    for rxn in sources:
        block_import(rxn)
    return [r.id for r in sources]


def biomass_reaction(model):
    coeffs = list(linear_reaction_coefficients(model).items())
    if len(coeffs) != 1 or coeffs[0][1] <= 0:
        raise ValueError(f"Expected one positive biomass objective; found {coeffs}")
    return coeffs[0][0]


def optimize_biomass(model, biomass):
    model.objective = biomass
    model.objective_direction = "max"
    result = model.optimize()
    return result.status, (result.fluxes[biomass.id] if result.status == "optimal" else None)


def no_external_nutrients(model, biomass):
    with model:
        for rxn in model.exchanges:
            block_import(rxn)
        status, flux = optimize_biomass(model, biomass)
    if status == "optimal" and flux is not None and flux <= TOL:
        return "PASS_zero_growth", flux
    if status == "optimal":
        return "FAIL_growth_without_external_nutrients", flux
    return f"REVIEW_{status}", None


def maximal_secretion_native(model, biomass, exchange, sources):
    """Maximize secretion in *original* model under anaerobic diet and >=10% growth."""
    with model:
        block_import(exchange)
        status, growth = optimize_biomass(model, biomass)
        if status != "optimal" or growth is None:
            return f"REVIEW_target_growth_{status}", growth, None, ""
        if growth <= TOL:
            return "NOT_EVALUABLE_no_growth_with_target_import_blocked", growth, None, ""
        biomass.lower_bound = max(biomass.lower_bound, GROWTH_FRACTION * growth)
        model.objective = exchange
        model.objective_direction = "max"
        sol = model.optimize()
        if sol.status != "optimal":
            return f"REVIEW_secretion_{sol.status}", growth, None, ""
        flux = float(sol.fluxes[exchange.id])
        used = ";".join(
            f"{r.id}:{sol.fluxes[r.id]:.6g}"
            for r in sources if abs(float(sol.fluxes[r.id])) > TOL
        )
        if flux > TOL and flux >= exchange.upper_bound - TOL:
            return "REVIEW_exchange_bound_reached", growth, flux, used
        if flux > TOL:
            return "MODEL_FEASIBLE_native_sources_not_experimental", growth, flux, used
        return "NO_SECRETION_FEASIBLE_under_tested_conditions", growth, flux, used


def individual_source_dependencies(model, biomass, sources):
    """Check whether individually blocking each pseudo-source impairs growth."""
    result = []
    for source in sources:
        with model:
            block_import(source)
            status, growth = optimize_biomass(model, biomass)
        result.append(f"{source.id}:{status}:{fmt(growth)}")
    return ";".join(result)


def no_nutrient_secretion(model, exchange):
    """Diagnostic for spontaneous secretion despite no extracellular inputs."""
    with model:
        for r in model.exchanges:
            block_import(r)
        model.objective = exchange
        model.objective_direction = "max"
        result = model.optimize()
        if result.status != "optimal":
            return f"REVIEW_{result.status}", None
        v = result.fluxes[exchange.id]
        return ("FAIL_spontaneous_secretion" if v > TOL else "PASS"), v


def fmt(value):
    return "" if value is None else f"{value:.9g}"


def main():
    for path in (ARCHIVE, DIET_PATH):
        if not path.is_file():
            raise SystemExit(f"Required input missing: {path}")
    diet = read_diet(DIET_PATH)
    digest = hashlib.sha256(DIET_PATH.read_bytes()).hexdigest()
    print(f"AGORA2 Western diet: {len(diet)} uptake entries; SHA256={digest}")
    print("Original-model secretion pilot; pseudo-source blocking only as an audit")
    records = []
    for mid in STRAINS:
        model, repaired_bytes = read_model(mid)
        biomass = biomass_reaction(model)
        sources = nonexchange_sources(model)
        matched = set_diet(model, diet)
        control, no_diet_growth = no_external_nutrients(model, biomass)
        growth_status, growth_diet = optimize_biomass(model, biomass)
        source_growth = individual_source_dependencies(model, biomass, sources)

        with model:
            for r in sources:
                block_import(r)
            strict_status, strict_growth = optimize_biomass(model, biomass)

        print(f"\n{mid}")
        print(f"  Diet entries matched: {matched}/{len(diet)}; encoding repair bytes: {repaired_bytes}")
        print(f"  Native growth on diet: {fmt(growth_diet)} ({growth_status}); no-diet: {fmt(no_diet_growth)} ({control})")
        print(f"  All pseudo-sources blocked: growth {fmt(strict_growth)} ({strict_status})")
        print("  One-at-a-time source block growth: " + (source_growth or "none"))
        if control != "PASS_zero_growth":
            print("  WARNING: Model fails no-diet growth control; all secretion results are not evaluable")

        for label, rid in TARGETS.items():
            blank_status = ""
            blank_flux = None
            target_growth = None
            max_flux = None
            source_fluxes = ""
            if rid not in {r.id for r in model.exchanges}:
                status = "EXCHANGE_ABSENT"
            else:
                exchange = model.reactions.get_by_id(rid)
                if boundary_coeff(exchange) >= 0:
                    status = "REVIEW_nonstandard_exchange_direction"
                elif control != "PASS_zero_growth":
                    status = "NOT_EVALUABLE_failed_no_diet_growth_control"
                elif growth_status != "optimal" or growth_diet is None or growth_diet <= TOL:
                    status = "NOT_EVALUABLE_no_growth_on_diet"
                else:
                    blank_status, blank_flux = no_nutrient_secretion(model, exchange)
                    if blank_status != "PASS":
                        status = "NOT_EVALUABLE_unexplained_secretion_without_diet"
                    else:
                        status, target_growth, max_flux, source_fluxes = maximal_secretion_native(
                            model, biomass, exchange, sources
                        )
            print(f"  {label:11s} {status:58s} max={fmt(max_flux)} blank={fmt(blank_flux)}")
            records.append({
                "Model_ID": mid,
                "Diet_SHA256": digest,
                "Diet_matched": matched,
                "Invalid_encoding_bytes": repaired_bytes,
                "Native_no_diet_growth_control": control,
                "Native_no_diet_growth": fmt(no_diet_growth),
                "Native_growth_on_diet": fmt(growth_diet),
                "Sources_in_model": ";".join(r.id for r in sources),
                "Individual_source_block_growth": source_growth,
                "All_sources_blocked_growth": fmt(strict_growth),
                "Metabolite": label,
                "Exchange_ID": rid,
                "No_diet_secretion_control": blank_status,
                "No_diet_max_secretion": fmt(blank_flux),
                "Max_growth_without_target_uptake": fmt(target_growth),
                "Min_growth_required": fmt(None if target_growth is None else GROWTH_FRACTION * target_growth),
                "Maximal_secretion_under_diet": fmt(max_flux),
                "Pseudo_source_fluxes_at_secretion_optimum": source_fluxes,
                "Interpretation": status,
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT.with_suffix(".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), delimiter="\t")
        w.writeheader()
        w.writerows(records)
    tmp_path.replace(OUTPUT)
    print(f"\nSaved only {OUTPUT}")
    print("These are conditional flux feasibility results, NOT measured production or validated IMPC indicators.")


if __name__ == "__main__":
    main()
