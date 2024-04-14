import torch
def Kmers_funct(seq, size=6):
    """
    Function to generate k-mers from a given sequence

    Parameters :
    ----------
    seq : str
            Sequence to be parsed
    size : int
            Size of the k-mer

    Returns :
    -------
    list
        List of k-mers
    """
    return [seq[x:x+size].upper() for x in range(len(seq) - size + 1)]

def tok_func(x): 
    """
    Function to tokenize the given sequence

    Parameters :
    ----------
    x : str
        Sequence to be tokenized

    Returns :
    -------
    str
        Tokenized sequence
    """
    return " ".join(Kmers_funct(x.replace("N","")))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        """
        Parameters :
        ----------
        encodings : dict
            Dictionary containing the tokenized sequences
        labels : list
            List of labels
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Parameters :
        ----------
        idx : int
            Index of the sequence to be retrieved

        Returns :
        -------
        dict
            Dictionary containing the tokenized sequence and the label
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])