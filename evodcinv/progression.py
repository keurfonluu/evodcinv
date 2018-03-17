# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

__all__ = [ "progress_bar", "progress_perc", "progress" ]


def progress_bar(i, imax, n = 50):
    bar = list("[" + n * " " + "]")
    perc = (i+1) / imax
    bar[1:int(perc*n)+1] = int(perc*n) * "="
    imid = (n+2) // 2
    if perc < 0.1:
        pstr = " %.2f%%" % (perc*100.)
    elif 0.1 <= perc < 1.:
        pstr = "%.2f%%" % (perc*100.)
    else:
        pstr = "100.0%"
    bar[imid-3:imid+3] = pstr
    print("\r" + "".join(bar), end = "", flush = True)
    
    
def progress_perc(i, imax, prefix = None):
    perc = (i+1) / imax
    if perc < 0.1:
        pstr = " %.2f%% " % (perc*100.)
    elif 0.1 <= perc < 1.:
        pstr = "%.2f%% " % (perc*100.)
    else:
        pstr = "100.0%"
    if prefix is None:
        prefix = "Progression: "
    print("\r%s%s" % (prefix, pstr), end = "", flush = True)
    
    
def progress(i, imax, ptype = "bar", n = 50, prefix = None):
    if ptype == "bar":
        progress_bar(i, imax, n)
    elif ptype == "perc":
        progress_perc(i, imax, prefix)
    else:
        raise ValueError("unknown progression type '%s'" % ptype)