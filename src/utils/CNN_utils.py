import numpy as np
def remove_N(seq):
    """
    Remove Ns from sequence

    Parameters
    ----------
    seq : str
        DNA sequence

    Returns
    -------
    str
        DNA sequence without Ns
    """
    return seq.upper().replace("N", "")

def onehote(seq):
    """
    One Hot encoding function

    Parameters
    ----------
    seq : str
        DNA sequence

    Returns
    -------
    np.array
        One hot encoded DNA sequence
    """
    seq2=list()
    mapping = {"A":[1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [0, 0., 1., 0.], "T":[0., 0., 0., 1.], "N":[0., 0., 0., 0.]}
    for i in seq:
      seq2.append(mapping[i]  if i in mapping.keys() else [0., 0., 0., 0.]) 
    return np.array(seq2)

def normalize_pwm(pwm, factor=3, max=None):
    """
    Function to normalize the PWM

    Parameters
    ----------
    pwm : np.array
        PWM matrix
    factor : int
        Factor to multiply the normalized PWM

    Returns
    -------
    np.array
        Normalized PWM
    """
	if not max:
		max = np.max(np.abs(pwm))
	pwm = pwm/max
	if factor:
		pwm = np.exp(pwm*factor)
	norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
	return pwm/norm