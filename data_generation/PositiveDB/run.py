def transform_name(name, genome_files):
    if "Taraxacum" in name:
         return "Taraxacum_kok-saghyz.genome.fasta.gz"
    elif "Aquilegia_coerulea" in name:
         return "Acoerulea_322_v3.fa.gz"
    elif "Ginkgo_biloba" in name:
         return "Gingko_biloba.1_0.fa.gz"
    elif "Cucurbita_maxima" in name:
         return "Cmaxima_genome_v1.1.fa.gz"
    elif "Cucurbita_pepo" in name:
         return "Cpepo_genome_v4.1.fa.gz"
    elif "Echinochloa_crus_galli" in name:
         return "Echinochloa_crus-galii.fasta.gz"
    elif "Cucurbita_moschata" in name:
         return "Cmoschata_genome_v1.fa.gz"
    elif "Solanum_lycopersicum" in name:
         return "S_lycopersicum_chromosomes.4.00.fa.gz"
    elif "Anthoceros_angustus" in name:
         return "Anthoceros.angustus.scaffold.fa.gz"
    else: 
         for g in genome_files:
                if name in g:
                     return g
from os import listdir
from os.path import isfile, join
genomes_path = "/mnt/extra/genomes/"
genomes_path2 = "/opt/genomes/"
annotation_path = "/opt/xhorvat9_TE_DBs/Genomes/Genomes/all_annotations/"
genome_files = [genomes_path + f for f in listdir(genomes_path) if isfile(join(genomes_path, f))] + [genomes_path2 + f for f in listdir(genomes_path2) if isfile(join(genomes_path2, f))]
annotation_files = [f for f in listdir(annotation_path) if isfile(join(annotation_path, f))]

def assign_annotation_file(genome_files, annotation_files):
    annot_dict = {}
    for a in annotation_files:
        name = a[:-4]
        # one exception
        annot_dict[name] = transform_name(name, genome_files)
    return annot_dict
annot_dict = assign_annotation_file(genome_files, annotation_files)

import tqdm
import pandas as pd
LTRs = {}
for species in annot_dict:
    
    print(f"Processing {species}")
    annot = pd.read_csv(f"{annotation_path}/{species}.txt", sep="\t", low_memory=False).iloc[:-1,]
    LTR_IDs = set(annot["LTR_ID"])

    LTRs[species] = {}

    for idx in tqdm.tqdm(LTR_IDs):
        element = annot[annot["LTR_ID"] == idx]
        element_5LTR = element[element["Domain"] == "intact_5ltr"][["Start", "End"]]
        element_3LTR = element[element["Domain"] == "intact_3ltr"][["Start", "End"]]
        info = element[element["Domain"] == "intact_5ltr"][["Superfamily", "Lineages", "Divergence", "Chromosome"]]
        try:
            superfamily = info.iloc[0, 0]
            lineage = info.iloc[0, 1]
            divergence = info.iloc[0, 2]
            chromosome = info.iloc[0, 3]

            LTRs[species][idx] = ({"Start": int(element_5LTR["Start"].iloc[0]),"End": int(element_5LTR["End"].iloc[0])}, # 5' LTR position
                                  {"Start": int(element_3LTR["Start"].iloc[0]),"End": int(element_3LTR["End"].iloc[0])}, # 3' LTR position
                                  superfamily,                                                                           # superfamily   
                                    lineage,                                                                             # lineage
                                    divergence,                                                                          # divergence of LTR sequences
                                    chromosome)                                                                          
        except:
            pass
import json
  

with open('LTR_locations_extra.txt', 'w') as convert_file:
     convert_file.write(json.dumps(LTRs))
