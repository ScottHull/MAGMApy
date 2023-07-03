from numpy import isnan

def get_K(df, species, temperature, phase="liquid"):
    # Fe3O4 must be treated differently
    if species == "Fe3O4" and phase == "liquid":  # Fe3O4 has a non A + B/T form
        exponent = (-4.385894544 * 10 ** -1) + ((4.3038155175436 * 10 ** 3) / temperature) - (
                3.1050205223386055 * 10 ** 6) / (
                           temperature ** 2)
        return 10 ** exponent
    # if "C" in df.columns and C is not nan, use a quadratic equation
    if "C" in df.columns and not isnan(df["C"][species]):
        A, B, C = float(df["A"][species]), float(df["B"][species]), float(df["C"][species])
        return 10 ** (A + (B / temperature) + (C / (temperature ** 2)))
    # else use a linear equation
    A, B = float(df["A"][species]), float(df["B"][species])
    return 10 ** (A + (B / temperature))
