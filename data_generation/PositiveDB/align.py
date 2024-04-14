from Bio import Align
import tqdm
import Bio.SeqIO as SeqIO
from os import listdir
from os.path import isfile, join
aligner = Align.PairwiseAligner()
aligner.mode = 'local'

LTR_files = [f[5:] for f in listdir("/home/xhorvat9/ltr_bert/DatabaseGeneration/PositiveDB/LTRs") if isfile(join("/home/xhorvat9/ltr_bert/DatabaseGeneration/PositiveDB/LTRs", f))]

seq_dict = {}
for f in tqdm.tqdm(LTR_files):
    seq_dict[f] = {}
    ltr5 = None
    ltr3 = None
    for rec in SeqIO.parse(f"/home/xhorvat9/ltr_bert/DatabaseGeneration/PositiveDB/LTRs/LTRs_{f}", "fasta"):
        if ltr5 is None:
            ltr5 = rec
        else:
            ltr3 = rec
            idx = "_".join(rec.id.split("_")[:-1])
            # align sequences
            try:
                alignments = aligner.align(str(ltr5.seq),str(ltr3.seq))
                alignment = alignments[0]
                seq_dict[f][idx] = (alignment.score/min(len(ltr3.seq),len(ltr5.seq)), ltr5.description.split()[-1])
                
                
            except IndexError:
                seq_dict[f][idx] = (0, ltr5.description.split()[-1])
                
            except ValueError:
                print("Zero length sequence", f, idx)
            # reset sequences
            ltr5 = None
            ltr3 = None

import json

with open('alignments.txt', 'w') as convert_file:
     convert_file.write(json.dumps(seq_dict))
