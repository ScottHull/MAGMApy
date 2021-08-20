def get_K(df, species, temperature):
    A, B = df["A"][species], df["B"][species]
    return 10 ** (A + (B / temperature))
