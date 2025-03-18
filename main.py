import pandas as pd
import numpy as np


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


# -----------------------------------------------------------------------------
# F) Exemple d'utilisation (main)
# -----------------------------------------------------------------------------
def main():
    # 1) Lecture des fichiers
    path_patchs = "../Donnees/spectres_patchs.xlsx"
    path_lights = "../Donnees/Lights_Telelumen.xlsx"
    path_cmf = "../Donnees/CMF_xyz_2deg.xlsx"

    df_patchs = lire_data_patchs(path_patchs)
    df_lights = lire_data_lights(path_lights)
    df_cmf = lire_data_cmf(path_cmf)

    # 2) Calcul XYZ de chaque éclairage (pour normaliser)
    df_xyz_ill = calcul_xyz_eclairage(df_lights, df_cmf)

    # 3) Calcul L*a*b* pour chaque (patch, éclairage)
    df_lab = calcul_lab_tous_patchs_eclairages(df_patchs, df_lights, df_cmf, df_xyz_ill)

    # 4) Enregistrement au format "tab-separated" (.tsv ou .txt)
    enregistrer_avec_tab(df_lab, "resultats_Lab.csv")


if __name__ == "__main__":
    main()
