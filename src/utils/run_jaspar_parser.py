import Bio.SeqIO as SeqIO
import pickle
from Bio import motifs
import sys
import tqdm
def get_motifs_list(motifs_file):
    """
    Returns array containing Bio.motifs.jaspar.Motif objects

    Parameters :
    ----------
    motifs_file : str
        Path to JASPAR motifs file

    Returns :
    -------
    list
        List of Bio.motifs.jaspar.Motif objects
    """
    with open(motifs_file, "r") as jspr:
        return motifs.parse(jspr, "jaspar")

def get_position_vector(sequence, jaspar_motifs):
    """
    Function parses the given Seq/str object using JASPAR profiles

    Parameters :
    ----------
    sequence : str
        Sequence to be parsed
    jaspar_motifs : list
        List of Bio.motifs.jaspar.Motif objects

    Returns : 
    -------
    dict 
        Dictionary with motif names as keys and list of positions as values
        
    """
    motif_positions = {motif.name: [] for motif in jaspar_motifs}
    for motif in jaspar_motifs:
        for pos, seq in motif.pssm.search(sequence, threshold=1, both=False):
            motif_positions[motif.name].append(pos)
    return motif_positions

def position_dict_to_occurence(positions_dict):
    """
    Converts the dictionary of positions to binary occurence value (1 if present, 0 if not)

    Parameters :
    ----------
    positions_dict : dict
        Dictionary with motif names as keys and list of positions as values
    
    Returns :
    -------
    dict
        Dictionary with motif names as keys and binary occurence as values
    """
    out_list = []
    for motif_name in positions_dict:
        if len(positions_dict[motif_name]) > 0:
            out_list.append(motif_name)
    return out_list

if len(sys.argv) < 4:
    print("Usage: python run_jaspar_parser.py <jaspar_file> <sequence_file> <output_file>")
    sys.exit(1)
    
jaspar_file = sys.argv[1]
#jaspar_file = "/storage/brno2/home/xhorvat9/Diplomovka_DBs/JASPAR_Profiles/All_Profiles_JASPAR.jaspar"
motifs = get_motifs_list(jaspar_file)
#seq_file = f"/storage/brno2/home/xhorvat9/ltr-annotator/Diplomovka_Final/1_Database_build/Negative_train_sequences/Negative_sequence_generation/all_length_non_LTRs_withMarkov{seq_num}.fasta"
seq_file = sys.argv[2]
seq_motif_list = {}

for rec in tqdm.tqdm(SeqIO.parse(seq_file, "fasta")):
    seq_motif_list[rec.id] = get_position_vector(rec.seq, motifs)
outfile = sys.argv[3]
pickle.dump(seq_motif_list, open(f"/var/tmp/xhorvat9/ltr_bert/Simple_ML_model/{outfile}", "wb+"))
