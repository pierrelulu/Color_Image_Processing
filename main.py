import pandas as pd
import numpy as np
import re

# -----------------------------------------------------------------------------
# A) Fonctions de lecture adaptées à VOS fichiers
# -----------------------------------------------------------------------------
def lire_data_patchs(path_patchs: str) -> pd.DataFrame:
    df = pd.read_excel(path_patchs, index_col=0)
    df.columns = df.columns.astype(float)
    return df


def lire_data_lights(path_lights: str) -> pd.DataFrame:
    df = pd.read_excel(path_lights, index_col=0)
    return df


def lire_data_cmf(path_cmf: str) -> pd.DataFrame:
    df = pd.read_excel(path_cmf)
    df.set_index("lambda", inplace=True)
    return df

def lire_data_rgb(path_rgb: str) -> pd.DataFrame:
    df = pd.read_excel(path_rgb)
    return df
# -----------------------------------------------------------------------------
# B) Calcul de XYZ pour chaque éclairage
# -----------------------------------------------------------------------------
def calcul_xyz_eclairage(df_lights: pd.DataFrame, df_cmf: pd.DataFrame) -> pd.DataFrame:
    eclairages = df_lights.columns
    results = {}
    for ecl in eclairages:
        common_waves = df_lights.index.intersection(df_cmf.index)
        E_vals = df_lights.loc[common_waves, ecl].values
        xbar = df_cmf.loc[common_waves, "x(lambda)"].values
        ybar = df_cmf.loc[common_waves, "y(lambda)"].values
        zbar = df_cmf.loc[common_waves, "z(lambda)"].values

        X_ill = np.trapz(E_vals * xbar, x=common_waves)
        Y_ill = np.trapz(E_vals * ybar, x=common_waves)
        Z_ill = np.trapz(E_vals * zbar, x=common_waves)
        results[ecl] = [X_ill, Y_ill, Z_ill]

    df_xyz_ill = pd.DataFrame.from_dict(results, orient='index', columns=["X_ill", "Y_ill", "Z_ill"])
    return df_xyz_ill


# -----------------------------------------------------------------------------
# C) Fonctions XYZ --> Lab et intégration patch × éclairage × CMF
# -----------------------------------------------------------------------------
def xyz_to_lab(xyz: np.array) -> np.array:
    def f(t):
        delta = 6 / 29
        return t ** (1 / 3) if t > delta ** 3 else (t / (3 * delta ** 2) + 4 / 29)

    X, Y, Z = xyz
    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))
    return np.array([L, a, b])

def calcul_xyz_patch_sous_eclairage(
        patch_name: str,
        df_patchs: pd.DataFrame,
        eclairage_name: str,
        df_lights: pd.DataFrame,
        df_cmf: pd.DataFrame,
        xyz_ill: np.array
) -> np.array:
    common_waves = df_patchs.columns.intersection(df_lights.index).intersection(df_cmf.index)
    if len(common_waves) == 0:
        return np.array([0.0, 0.0, 0.0])  # ou lever une exception

    refl_vals = df_patchs.loc[patch_name, common_waves].values
    light_vals = df_lights.loc[common_waves, eclairage_name].values
    xbar = df_cmf.loc[common_waves, "x(lambda)"].values
    ybar = df_cmf.loc[common_waves, "y(lambda)"].values
    zbar = df_cmf.loc[common_waves, "z(lambda)"].values

    X_val = np.trapz(refl_vals * light_vals * xbar, x=common_waves)
    Y_val = np.trapz(refl_vals * light_vals * ybar, x=common_waves)
    Z_val = np.trapz(refl_vals * light_vals * zbar, x=common_waves)

    X_ill, Y_ill, Z_ill = xyz_ill
    X_n = X_val / X_ill if X_ill != 0 else 0
    Y_n = Y_val / Y_ill if Y_ill != 0 else 0
    Z_n = Z_val / Z_ill if Z_ill != 0 else 0

    return np.array([X_n, Y_n, Z_n])

# -----------------------------------------------------------------------------
# C bis) Fonctions XYZ --> sRGB et intégration patch × éclairage
# -----------------------------------------------------------------------------
def xyz_to_srgb(xyz: np.array) -> np.array:
    """
    Convertit les valeurs XYZ en valeurs RGB en utilisant la matrice de conversion standard sRGB.

    Paramètres
    ----------
    xyz : np.array
        Valeurs XYZ à convertir.

    Retourne
    --------
    np.array
        Valeurs RGB correspondantes.
    """
    # Matrice de conversion XYZ vers sRGB
    matrice_conversion = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    # Appliquer la conversion
    rgb = np.dot(matrice_conversion, xyz)

    # Appliquer une correction gamma pour sRGB
    rgb = np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(rgb, 1/2.4) - 0.055
    )
    # Normaliser les valeurs RGB pour qu'elles soient dans la plage [0, 255]
    rgb = np.clip(rgb * 255, 0, 255)
    return rgb

def calcul_rgb_patch_sous_eclairage(
        patch_name: str,
        df_patchs: pd.DataFrame,
        eclairage_name: str,
        df_lights: pd.DataFrame,
        df_cmf: pd.DataFrame,
        df_xyz_ill: np.array
) -> np.array:
    """
    Calcule les valeurs RGB d'un patch sous un éclairage donné.

    Paramètres
    ----------
    patch_name : str
        Nom du patch.
    df_patchs : pd.DataFrame
        DataFrame contenant les données spectrales des patchs.
    eclairage_name : str
        Nom de l'éclairage.
    df_lights : pd.DataFrame
        DataFrame contenant les données des éclairages.
    df_cmf : pd.DataFrame
        DataFrame contenant les fonctions colorimétriques.
    xyz_ill : np.array
        Valeurs XYZ de l'illuminant.

    Retourne
    --------
    np.array
        Valeurs RGB du patch sous l'éclairage donné.
    """
    # Calculer les valeurs XYZ du patch sous l'éclairage
    xyz_ill = df_xyz_ill.loc[eclairage_name].values
    xyz_patch = calcul_xyz_patch_sous_eclairage(
        patch_name, df_patchs, eclairage_name, df_lights, df_cmf, xyz_ill
    )

    # Convertir les valeurs XYZ en RGB
    rgb_patch = xyz_to_srgb(xyz_patch)

    return rgb_patch

# -----------------------------------------------------------------------------
# D) Calcul du Lab pour tous les (patch, éclairage)
# -----------------------------------------------------------------------------
def calcul_lab_tous_patchs_eclairages(
        df_patchs: pd.DataFrame,
        df_lights: pd.DataFrame,
        df_cmf: pd.DataFrame,
        df_xyz_ill: pd.DataFrame
) -> pd.DataFrame:
    rows_result = []
    all_patches = df_patchs.index
    all_eclairages = df_lights.columns

    for patch_name in all_patches:
        for ecl_name in all_eclairages:
            xyz_ill = df_xyz_ill.loc[ecl_name].values
            xyz_patch = calcul_xyz_patch_sous_eclairage(
                patch_name, df_patchs, ecl_name, df_lights, df_cmf, xyz_ill
            )
            L_val, a_val, b_val = xyz_to_lab(xyz_patch)
            rows_result.append({
                "Patch": patch_name,
                "Eclairage": ecl_name,
                "L": L_val,
                "a": a_val,
                "b": b_val
            })
    return pd.DataFrame(rows_result)


# -----------------------------------------------------------------------------
# E) Nouveau : fonction d'enregistrement avec séparateur tab
# -----------------------------------------------------------------------------
def enregistrer_avec_tab(df: pd.DataFrame, filename: str) -> None:
    """
    Enregistre le DataFrame avec un séparateur tab (\t) pour respecter
    la mise en forme de type 'tsv'.
    """
    df.to_csv(filename, index=False, sep=';')
    print(f"Fichier enregistré avec séparateur tab : {filename}")


from itertools import combinations

def calculer_distances_lab(df_lab: pd.DataFrame) -> pd.DataFrame:
    df_lab['Patch'] = df_lab['Patch'].astype(str)
    df_lab['Eclairage'] = df_lab['Eclairage'].astype(str)

    distances = []
    for ecl_name, group in df_lab.groupby('Eclairage'):
        patch_coords = {
            row.Patch: np.array([row.L, row.a, row.b])
            for row in group.itertuples()
        }
        for (p1, v1), (p2, v2) in combinations(patch_coords.items(), 2):
            dist = np.linalg.norm(v1 - v2)
            distances.append({
                "Eclairage": ecl_name,
                "Patch1": p1,
                "Patch2": p2,
                "Distance": dist
            })

    return pd.DataFrame(distances)


# -----------------------------------------------------------------------------
# H) Recherche du couple de patchs optimal
# -----------------------------------------------------------------------------
def trouver_couple_optimal(df_distances: pd.DataFrame):
    minima = df_distances.loc[df_distances.groupby("Eclairage")["Distance"].idxmin()]
    maxima = df_distances.loc[df_distances.groupby("Eclairage")["Distance"].idxmax()]

    min_set = {(row.Patch1, row.Patch2) for row in minima.itertuples()}
    max_set = {(row.Patch1, row.Patch2) for row in maxima.itertuples()}
    intersection = min_set & max_set
    if intersection:
        couple = list(intersection)[0]
    else:
        # Trouver le couple avec le plus grand écart (distance max - distance min)
        candidats = []
        for row_min in minima.itertuples():
            for row_max in maxima.itertuples():
                if {row_min.Patch1, row_min.Patch2} == {row_max.Patch1, row_max.Patch2}:
                    ecart = row_max.Distance - row_min.Distance
                    candidats.append((ecart, row_min.Patch1, row_min.Patch2))
        if candidats:
            candidats.sort(reverse=True)
            couple = (candidats[0][1], candidats[0][2])
        else:
            couple = None

    if couple:
        ecl_min = minima[
            ((minima["Patch1"] == couple[0]) & (minima["Patch2"] == couple[1])) |
            ((minima["Patch1"] == couple[1]) & (minima["Patch2"] == couple[0]))
        ]["Eclairage"].tolist()

        ecl_max = maxima[
            ((maxima["Patch1"] == couple[0]) & (maxima["Patch2"] == couple[1])) |
            ((maxima["Patch1"] == couple[1]) & (maxima["Patch2"] == couple[0]))
        ]["Eclairage"].tolist()

        return couple, ecl_min, ecl_max
    else:
        return None, [], []


def trouver_couple_optimal_par_ecart(df_distances: pd.DataFrame):
    """
    Pour chaque couple (Patch1, Patch2), on calcule la distance Lab sous
    tous les éclairages. On cherche l'écart (distance max - distance min).
    On retourne le couple qui a l'écart le plus grand, et les éclairages
    où se produisent ce min et ce max.
    """
    # Pour éviter les problèmes d'ordre (Patch1, Patch2) vs (Patch2, Patch1),
    # on peut normaliser dans une colonne "Pair" :
    df_temp = df_distances.copy()
    df_temp['Pair'] = df_temp.apply(
        lambda row: tuple(sorted([row['Patch1'], row['Patch2']])), axis=1
    )

    # Groupement par couple
    grouped = df_temp.groupby('Pair')

    best_pair = None
    best_ecart = -1
    best_ecl_min = []
    best_ecl_max = []

    for pair, subdf in grouped:
        # Distance min et max de ce couple, tous éclairages confondus
        dist_min = subdf['Distance'].min()
        dist_max = subdf['Distance'].max()
        ecart = dist_max - dist_min

        if ecart > best_ecart:
            best_ecart = ecart
            best_pair = pair
            # Quels éclairages donnent cette distance min ?
            best_ecl_min = subdf[subdf['Distance'] == dist_min]['Eclairage'].unique().tolist()
            # Quels éclairages donnent cette distance max ?
            best_ecl_max = subdf[subdf['Distance'] == dist_max]['Eclairage'].unique().tolist()

    return best_pair, best_ecl_min, best_ecl_max


def trouver_couple_optimal_avec_contrainte(df_distances: pd.DataFrame, seuil_proche_zero: float = 2.0):
    """
    Cherche le couple (Patch1, Patch2) dont la distance MIN (sous un certain éclairage)
    est < seuil_proche_zero, et qui maximise l'écart (distance_max - distance_min).

    Paramètres
    ----------
    df_distances : DataFrame contenant les colonnes [Eclairage, Patch1, Patch2, Distance]
    seuil_proche_zero : float, distance en dessous de laquelle on considère les patchs "quasi identiques"

    Retourne
    --------
    best_pair : tuple (PatchA, PatchB)
    best_min_dist : float, distance minimale sur au moins un éclairage pour ce couple
    best_max_dist : float, distance maximale sur un autre éclairage pour ce couple
    best_ecl_min : list of str, éclairage(s) où la distance est min
    best_ecl_max : list of str, éclairage(s) où la distance est max
    """

    # (Patch1, Patch2) peut apparaître dans l'ordre ou l'inverse dans df_distances,
    # on unifie en créant une colonne "Pair" avec (PatchX, PatchY) dans l'ordre trié.
    df_temp = df_distances.copy()
    df_temp['Pair'] = df_temp.apply(
        lambda row: tuple(sorted([row['Patch1'], row['Patch2']])), axis=1
    )

    # On regroupe par ce couple unifié
    grouped = df_temp.groupby('Pair')

    best_pair = None
    best_ecart = -1  # on veut maximiser l'écart = (max_dist - min_dist)
    best_min_dist = None
    best_max_dist = None
    best_ecl_min = []
    best_ecl_max = []

    for pair, subdf in grouped:
        # subdf contient toutes les lignes (éclairages) pour ce couple
        dist_min = subdf['Distance'].min()
        dist_max = subdf['Distance'].max()

        # On ne s'intéresse à ce couple que si sa distance min < seuil_proche_zero
        if dist_min < seuil_proche_zero:
            ecart = dist_max - dist_min
            if ecart > best_ecart:
                best_ecart = ecart
                best_pair = pair
                best_min_dist = dist_min
                best_max_dist = dist_max

                # Quels éclairages donnent la distance min ?
                best_ecl_min = subdf[subdf['Distance'] == dist_min]['Eclairage'].unique().tolist()
                # Quels éclairages donnent la distance max ?
                best_ecl_max = subdf[subdf['Distance'] == dist_max]['Eclairage'].unique().tolist()

    return best_pair, best_min_dist, best_max_dist, best_ecl_min, best_ecl_max

def trouver_couple_optimal_par_prox(df_distances: pd.DataFrame, seuil_loin_zero: float = 5.0):
    """
    Cherche le couple (Patch1, Patch2) dont la distance MIN (sous un certain éclairage)
    est minimale, et dont la distance_max est supérieure à seuil_loin_zero.

    Paramètres
    ----------
    df_distances : DataFrame contenant les colonnes [Eclairage, Patch1, Patch2, Distance]
    seuil_proche_zero : float, distance en dessous de laquelle on considère les patchs "quasi identiques"

    Retourne
    --------
    best_pair : tuple (PatchA, PatchB)
    best_min_dist : float, distance minimale sur au moins un éclairage pour ce couple
    best_max_dist : float, distance maximale sur un autre éclairage pour ce couple
    best_ecl_min : list of str, éclairage(s) où la distance est min
    best_ecl_max : list of str, éclairage(s) où la distance est max
    """

    # (Patch1, Patch2) peut apparaître dans l'ordre ou l'inverse dans df_distances,
    # on unifie en créant une colonne "Pair" avec (PatchX, PatchY) dans l'ordre trié.
    df_temp = df_distances.copy()
    df_temp['Pair'] = df_temp.apply(
        lambda row: tuple(sorted([row['Patch1'], row['Patch2']])), axis=1
    )

    # On regroupe par ce couple unifié
    grouped = df_temp.groupby('Pair')

    best_pair = None
    best_min_dist = 1e6 # on veut minimiser dE pour les couple suffisemment differents
    best_max_dist = None
    best_ecl_min = []
    best_ecl_max = []

    for pair, subdf in grouped:
        # subdf contient toutes les lignes (éclairages) pour ce couple
        dist_min = subdf['Distance'].min()
        dist_max = subdf['Distance'].max()

        # On ne s'intéresse à ce couple que si sa distance max > seuil_loin_zero
        if dist_max > seuil_loin_zero:
            if dist_min < best_min_dist:
                best_pair = pair
                best_min_dist = dist_min
                best_max_dist = dist_max

                # Quels éclairages donnent la distance min ?
                best_ecl_min = subdf[subdf['Distance'] == dist_min]['Eclairage'].unique().tolist()
                # Quels éclairages donnent la distance max ?
                best_ecl_max = subdf[subdf['Distance'] == dist_max]['Eclairage'].unique().tolist()

    return best_pair, best_min_dist, best_max_dist, best_ecl_min, best_ecl_max

def convsersion_patch_rgb(nom_patch: str, df_rgb: pd.DataFrame):
    """
    Obtient les valeurs RGB d'un patch à partir de son nom.

    Paramètres
    ----------
    nom_patch : str
        Nom du patch sous la forme "Patch XXX".
    df_rgb : pd.DataFrame
        DataFrame contenant les valeurs RGB des patchs, indexé par le numéro de patch.

    Retourne
    --------
    np.array
        Valeurs RGB correspondantes.
    """
    # Extraire le numéro du patch à partir du nom
    match = re.match(r"Patch (\d+)", nom_patch)
    if not match:
        raise ValueError(f"Le nom du patch '{nom_patch}' n'est pas au bon format.")

    # Obtenir le numéro du patch
    numero_patch = int(match.group(1))

    # Accéder aux valeurs RGB dans le DataFrame
    if numero_patch in df_rgb.index:
        return 255*df_rgb.loc[numero_patch].values
    else:
        raise KeyError(f"Le patch {numero_patch} n'existe pas dans df_rgb.")

# -----------------------------------------------------------------------------
# F) Exemple d'utilisation (main)
# -----------------------------------------------------------------------------
def main():
    # 1) Lecture des fichiers
    path_patchs = "../Donnees/spectres_patchs.xlsx"
    path_lights = "../Donnees/Lights_Telelumen.xlsx"
    path_cmf = "../Donnees/CMF_xyz_2deg.xlsx"
    path_rgb = "../Donnees/RGB.xlsx"

    df_patchs = lire_data_patchs(path_patchs)
    df_lights = lire_data_lights(path_lights)
    df_cmf = lire_data_cmf(path_cmf)
    df_rgb = lire_data_rgb(path_rgb)

    # 2) Calcul XYZ de chaque éclairage
    df_xyz_ill = calcul_xyz_eclairage(df_lights, df_cmf)

    # 3) Calcul Lab pour tous les (patch, éclairage)
    df_lab = calcul_lab_tous_patchs_eclairages(df_patchs, df_lights, df_cmf, df_xyz_ill)

    # 4) Enregistrement des Lab
    enregistrer_avec_tab(df_lab, "resultats_Lab.csv")

    # 5) Calcul des distances entre patchs pour chaque éclairage
    df_distances = calculer_distances_lab(df_lab)

    # 6) Recherche du couple de patchs optimal en imposant la contrainte
    seuil = 100# ou 1.0 ou 0.5, selon votre tolérance
    (couple, dist_min, dist_max, list_ecl_min, list_ecl_max
    ) = trouver_couple_optimal_par_prox(df_distances, seuil)

    if couple is None:
        print(f"Aucun couple de patchs n'a une distance min < {seuil}.")
        return

    ecl_max = list_ecl_max[0]
    ecl_min = list_ecl_min[0]
    patchA, patchB = couple
    print(f"\nCouple de patchs optimal : {couple}")
    print(f"Distance MIN = {dist_min:.3f}, sous l'éclairage : {ecl_min}")
    print(f"Distance MAX = {dist_max:.3f}, sous l'éclairage : {ecl_max}")
    print(f"Écart = {dist_max - dist_min:.3f}")
    print(f"{patchA} : {convsersion_patch_rgb(patchA, df_rgb)} ; {patchA} : {convsersion_patch_rgb(patchB, df_rgb)}")

    #Calcul des valeurs sRGB pour les patchs sous les 2 illuminants
    A_mindiff = calcul_rgb_patch_sous_eclairage(
        patchA, df_patchs, ecl_min, df_lights, df_cmf, df_xyz_ill
    )
    B_mindiff = calcul_rgb_patch_sous_eclairage(
        patchB, df_patchs, ecl_min, df_lights, df_cmf, df_xyz_ill
    )
    A_maxdiff = calcul_rgb_patch_sous_eclairage(
        patchA, df_patchs, ecl_max, df_lights, df_cmf, df_xyz_ill
    )
    B_maxdiff = calcul_rgb_patch_sous_eclairage(
        patchB, df_patchs, ecl_max, df_lights, df_cmf, df_xyz_ill
    )
    print("sRGB des patchs sous les éclairages min et max :")
    print(f"{ecl_min}: {A_mindiff} // {B_mindiff}\n{ecl_max}: {A_maxdiff} // {B_maxdiff}")

if __name__ == "__main__":
    main()
