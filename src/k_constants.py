def get_K(df, species, temperature):
    A, B = float(df["A"][species]), float(df["B"][species])
    return 10 ** (A + (B / temperature))
