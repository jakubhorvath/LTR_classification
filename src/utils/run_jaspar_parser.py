import Bio.SeqIO as SeqIO
import pickle
from Bio import motifs
import sys
import tqdm
import argparse
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
    
parser = argparse.ArgumentParser()
parser.add_argument('--jaspar_file', help='Path to jaspar db')
parser.add_argument('--seq_file', help='Path to sequence file')
parser.add_argument('--out_file', help='Path to out directory')
args = parser.parse_args()

jaspar_file = args.jaspar_file
motifs = get_motifs_list(jaspar_file)
seq_file = args.seq_file
seq_motif_list = {}

for rec in tqdm.tqdm(SeqIO.parse(seq_file, "fasta")):
    seq_motif_list[rec.id] = get_position_vector(rec.seq, motifs)
outfile = args.out_file
pickle.dump(seq_motif_list, open(f"{outfile}/TFBS_occurences.b", "wb+"))
