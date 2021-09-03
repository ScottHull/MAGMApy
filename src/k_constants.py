def get_K(df, species, temperature, phase="liquid"):
    if species == "Fe3O4" and phase == "liquid":  # Fe3O4 has a non A + B/T form
        exponent = (-4.385894544 * 10 ** -1) + ((4.3038155175436 * 10 ** 3) / temperature) - (
                3.1050205223386055 * 10 ** 6) / (
                           temperature ** 2)
        return 10 ** exponent
    A, B = float(df["A"][species]), float(df["B"][species])
    return 10 ** (A + (B / temperature))
