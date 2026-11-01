from pathlib import Path
from collections import defaultdict, Counter
import csv
import io
import subprocess
import re
import zipfile
import xml.etree.ElementTree as ET

folder = Path('/home/ibab/PycharmProjects/Autism_gut_microbiome/15_10')
archive = folder / 'AGORA2_SBML.zip'
mapping = folder / 'AGORA2_Baseline_XGBoost_Top50_species_model_ids.tsv'
output = folder / 'AGORA2_27_species_analysis' / '02_species_metabolite_mapping.tsv'


def local_name(tag):
    return tag.rsplit('}', 1)[-1]


def cobra_id(raw, prefix):
    raw = re.sub(r'__(\d+)__', lambda m: chr(int(m.group(1))), raw)
    return raw.removeprefix(prefix)


def repair_utf8_if_needed(xml_bytes, model_id):
    """Recode only invalid UTF-8 bytes using cp1252; never alter valid UTF-8.

    Mixed-encoding metadata remains scientifically ambiguous: repairs are logged.
    If a byte has no cp1252 mapping, stop instead of silently dropping it.
    """
    try:
        xml_bytes.decode('utf-8')
        return xml_bytes, False
    except UnicodeDecodeError:
        decoded = xml_bytes.decode('utf-8', errors='surrogateescape')
        counts = Counter(ord(ch) - 0xDC00 for ch in decoded
                         if 0xDC80 <= ord(ch) <= 0xDCFF)
        if not counts:
            raise ValueError(f'Unexpected UTF-8 decoding error in {model_id}')
        replacements = {}
        for byte in counts:
            try:
                replacements[0xDC00 + byte] = bytes([byte]).decode('cp1252')
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f'{model_id}: invalid byte 0x{byte:02X} has no cp1252 mapping; '
                    'cannot safely repair automatically'
                ) from exc
        repaired_text = ''.join(replacements.get(ord(ch), ch) for ch in decoded)
        repaired_bytes = repaired_text.encode('utf-8')
        description = ', '.join(f'0x{b:02X} x{n}' for b, n in sorted(counts.items()))
        print(f'WARNING {model_id}: recoded invalid UTF-8 byte(s) as cp1252: '
              f'{description}. Check affected metabolite names.', flush=True)
        return repaired_bytes, True


def exchanges_from_xml(xml_bytes, model_id):
    xml_bytes, encoding_repaired = repair_utf8_if_needed(xml_bytes, model_id)
    metabolites = {}
    exchanges = set()
    # Parse each model in memory; no extracted XML files are saved.
    for _, element in ET.iterparse(io.BytesIO(xml_bytes), events=('end',)):
        tag = local_name(element.tag)
        if tag == 'species':
            metabolites[element.get('id')] = (
                element.get('name', ''), element.get('compartment', '')
            )
            element.clear()
        elif tag == 'reaction':
            reaction_id = cobra_id(element.get('id', ''), 'R_')
            if reaction_id.startswith('EX_'):
                refs = set()
                for part in element:
                    if local_name(part.tag) in ('listOfReactants', 'listOfProducts'):
                        for ref in part:
                            if local_name(ref.tag) == 'speciesReference':
                                refs.add(ref.get('species'))
                if len(refs) == 1:
                    raw_met = next(iter(refs))
                    if raw_met in metabolites:
                        name, compartment = metabolites[raw_met]
                        if (compartment.removeprefix('C_') in ('e', 'extracellular')
                                or cobra_id(raw_met, 'M_').endswith('[e]')):
                            met_id = cobra_id(raw_met, 'M_')
                            exchanges.add((met_id, name or met_id, reaction_id))
            element.clear()
    return exchanges, encoding_repaired


def main():
    for path in (archive, mapping):
        if not path.is_file():
            raise SystemExit(f'Missing file: {path}')
    with mapping.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file, delimiter='\t'))
    required = {'Baseline_Rank', 'Input_Taxon', 'AGORA2_Species', 'AGORA2_MicrobeID'}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit('Invalid or empty mapping file')

    species_counts = defaultdict(int)
    findings = {}
    repaired_models = 0

    with zipfile.ZipFile(archive) as z:
        member_names = {Path(name).stem: name for name in z.namelist() if name.endswith('.xml')}
        total = len(rows)
        for i, row in enumerate(rows, start=1):
            model_id = row['AGORA2_MicrobeID'].strip()
            member = member_names.get(model_id)
            if member is None:
                raise SystemExit(f'XML missing from ZIP: {model_id}')
            taxon_key = (int(row['Baseline_Rank']), row['Input_Taxon'], row['AGORA2_Species'])

            try:
                try:
                    xml_bytes = z.read(member)
                except NotImplementedError:
                    xml_bytes = subprocess.run(
                        ['unzip', '-p', str(archive), member],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
                    ).stdout
                exchanges, repaired = exchanges_from_xml(xml_bytes, model_id)
            except Exception as e:
                raise SystemExit(f'Failed to parse {model_id}: {e}') from e

            if not exchanges:
                raise SystemExit(f'No extracellular EX_ reactions found: {model_id}. '
                                 'Stop and inspect XML format.')

            species_counts[taxon_key] += 1
            repaired_models += int(repaired)
            per_metabolite = defaultdict(set)
            names = {}
            for met_id, met_name, reaction_id in exchanges:
                per_metabolite[met_id].add(reaction_id)
                names[met_id] = met_name
            for met_id, reaction_ids in per_metabolite.items():
                key = (*taxon_key, met_id)
                if key not in findings:
                    findings[key] = {'name': names[met_id], 'models': 0,
                                     'reactions': set(), 'repaired': 0}
                findings[key]['models'] += 1
                findings[key]['reactions'].update(reaction_ids)
                findings[key]['repaired'] += int(repaired)

            if i % 50 == 0 or i == total:
                print(f'Processed {i}/{total} models', flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Baseline_Rank', 'Input_Taxon', 'AGORA2_Species',
                         'Metabolite_ID', 'Metabolite_Name', 'Exchange_Reaction_IDs',
                         'Strains_with_exchange', 'Strains_evaluated',
                         'Fraction_with_exchange', 'Models_with_encoding_repair'])
        for key, info in sorted(findings.items(), key=lambda pair: (pair[0][0], pair[0][3])):
            rank, taxon, species, met_id = key
            n = species_counts[(rank, taxon, species)]
            writer.writerow([rank, taxon, species, met_id, info['name'],
                             ';'.join(sorted(info['reactions'])), info['models'], n,
                             round(info['models'] / n, 4), info['repaired']])
    print(f'\nSpecies evaluated: {len(species_counts)}')
    print(f'Models evaluated: {sum(species_counts.values())}')
    print(f'Models requiring encoding repair: {repaired_models}')
    print(f'Species-metabolite rows: {len(findings)}')
    print(f'One output: {output}')
    print('Exchange presence ONLY; production/secretion has NOT been demonstrated.')


if __name__ == '__main__':
    main()
