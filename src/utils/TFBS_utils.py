
def get_presence_count_dict(motif_dict_count, motif_dict_presence, TF_sites):
    """
    Function to generate a dictionary of motif presence and count

    Parameters :
    ----------
    motif_dict_count : dict
        Dictionary to store motif count
    motif_dict_presence : dict
        Dictionary to store motif presence
    TF_sites : dict
        Dictionary containing TF binding sites

    Returns :
    -------
    None
    """
    for seq in TF_sites:
        for motif in motif_dict_count:
            if len(TF_sites[seq][motif]) > 0:
                motif_dict_count[motif].append(len(TF_sites[seq][motif]))
                motif_dict_presence[motif].append(1)
            else:
                motif_dict_count[motif].append(0)
                motif_dict_presence[motif].append(0)