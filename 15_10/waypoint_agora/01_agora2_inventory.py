from pathlib import Path
import csv
import zipfile
from collections import defaultdict

from pathlib import Path

folder = Path("/home/ibab/PycharmProjects/Autism_gut_microbiome/15_10")

archive = folder / "AGORA2_SBML.zip"
mapping = folder / "AGORA2_Baseline_XGBoost_Top50_species_model_ids.tsv"
outdir = folder / "AGORA2_27_species_analysis"

for path in (archive, mapping):
    if not path.is_file():
        raise SystemExit(f'Missing file: {path}\nPlace this file in ~/Downloads and run again.')

with mapping.open(newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f, delimiter='\t'))

needed = {'Baseline_Rank', 'Input_Taxon', 'AGORA2_MicrobeID', 'AGORA2_Species'}
if not rows or not needed.issubset(rows[0]):
    raise SystemExit('Mapping TSV is empty or missing required columns.')

with zipfile.ZipFile(archive) as z:
    xml_members = [n for n in z.namelist() if n.lower().endswith('.xml') and not n.endswith('/')]
    filenames = defaultdict(list)
    for name in xml_members:
        filenames[Path(name).stem].append(name)

outdir.mkdir(parents=True,exist_ok=True)
inventory = outdir / '01_model_inventory.tsv'
summary_file = outdir / '01_species_summary.tsv'

results = []
by_species = defaultdict(lambda: {'rank': '', 'species': '', 'expected': 0, 'found': 0})
for row in rows:
    model_id = row['AGORA2_MicrobeID'].strip()
    source_taxon = row['Input_Taxon'].strip()
    key = (row['Baseline_Rank'], source_taxon)
    matches = filenames.get(model_id, [])
    rec = by_species[key]
    rec['rank'] = row['Baseline_Rank']
    rec['species'] = row['AGORA2_Species']
    rec['expected'] += 1
    rec['found'] += bool(matches)
    results.append({
        'Baseline_Rank': row['Baseline_Rank'],
        'Input_Taxon': source_taxon,
        'AGORA2_Species': row['AGORA2_Species'],
        'AGORA2_MicrobeID': model_id,
        'In_ZIP': 'YES' if matches else 'NO',
        'ZIP_member': matches[0] if matches else '',
    })

with inventory.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0]), delimiter='\t')
    w.writeheader()
    w.writerows(results)

with summary_file.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['Baseline_Rank', 'Input_Taxon', 'AGORA2_Species', 'Models_expected', 'Models_in_ZIP', 'Models_missing'])
    for (rank, taxon), data in sorted(by_species.items(), key=lambda kv: int(kv[0][0])):
        missing = data['expected'] - data['found']
        w.writerow([rank, taxon, data['species'], data['expected'], data['found'], missing])
        print(f'{int(rank):2d}  {taxon:46s} {data["found"]:4d}/{data["expected"]:4d} models')

print(f'\nSpecies: {len(by_species)}')
print(f'Model IDs in mapping: {len(results)}')
print(f'Model files located: {sum(r["In_ZIP"] == "YES" for r in results)}')
print(f'Output: {inventory}')
print(f'Output: {summary_file}')
print('No metabolic production claims made: this is an inventory only.')
