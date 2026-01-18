"""
Sifat fisikokimia 20 asam amino standar
Digunakan untuk ekstraksi fitur dari sekuens protein HA
"""

# Hydrophobicity (Kyte-Doolittle scale)
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Volume (Å³)
VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
    'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
}

# Polarity
POLARITY = {
    'A': 0.0, 'R': 52.0, 'N': 3.38, 'D': 40.7, 'C': 1.48,
    'Q': 3.53, 'E': 49.91, 'G': 0.0, 'H': 51.6, 'I': 0.13,
    'L': 0.13, 'K': 49.5, 'M': 1.43, 'F': 0.35, 'P': 1.58,
    'S': 1.67, 'T': 1.66, 'W': 2.1, 'Y': 1.61, 'V': 0.13
}


# Charge at pH 7
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

# Molecular Weight (Da)
MOLECULAR_WEIGHT = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
}

# Isoelectric Point (pI)
ISOELECTRIC_POINT = {
    'A': 6.0, 'R': 10.8, 'N': 5.4, 'D': 2.8, 'C': 5.1,
    'Q': 5.7, 'E': 3.2, 'G': 6.0, 'H': 7.6, 'I': 6.0,
    'L': 6.0, 'K': 9.7, 'M': 5.7, 'F': 5.5, 'P': 6.3,
    'S': 5.7, 'T': 5.6, 'W': 5.9, 'Y': 5.7, 'V': 6.0
}

# Gabungan semua properties
ALL_PROPERTIES = {
    'hydrophobicity': HYDROPHOBICITY,
    'volume': VOLUME,
    'polarity': POLARITY,
    'charge': CHARGE,
    'molecular_weight': MOLECULAR_WEIGHT,
    'isoelectric_point': ISOELECTRIC_POINT
}

# Epitope sites pada HA H3N2 (posisi berbasis 0)
EPITOPE_SITES = {
    'A': list(range(122, 147)),
    'B': list(range(155, 196)),
    'C': list(range(50, 54)) + list(range(275, 280)),
    'D': list(range(172, 182)) + list(range(201, 220)),
    'E': list(range(62, 65)) + list(range(78, 84))
}