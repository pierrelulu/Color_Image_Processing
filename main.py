import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# 1. Lecture et préparation des données
# =============================================================================
def lire_data_batch(path_batch: str) -> pd.DataFrame:
    """
    Lit le fichier Excel contenant les spectres ou les informations
    nécessaires sur les batchs (patchs).
    """
    df_batch = pd.read_excel(path_batch)  # Ajuster les options de lecture si nécessaire
    # Traiter ou renommer des colonnes si besoin
    return df_batch


def lire_data_cmf(path_cmf: str) -> pd.DataFrame:
    """
    Lit le fichier Excel contenant les fonctions colorimétriques CMF
    (x̅, y̅, z̅) pour l’observateur 2°.
    """
    df_cmf = pd.read_excel(path_cmf)  # Ajuster les options de lecture si nécessaire
    # On s'attend à y trouver au moins : [Longueur d'onde, x_bar, y_bar, z_bar]
    return df_cmf


def lire_data_eclairage(path_lights: str) -> pd.DataFrame:
    """
    Lit le fichier Excel contenant les spectres (SPD) des différents éclairages.
    """
    df_lights = pd.read_excel(path_lights)  # Ajuster les options de lecture si nécessaire
    # On s'attend à y trouver : [Longueur d'onde, Eclairage_1, Eclairage_2, ...]
    return df_lights


def lire_data_rgb(path_rgb: str) -> pd.DataFrame:
    """
    Lit le fichier Excel contenant d'éventuelles correspondances ou conversions vers le RGB.
    Par exemple un tableau qui mappe L*a*b* -> R, G, B ou XYZ -> R, G, B.
    """
    df_rgb = pd.read_excel(path_rgb)  # Ajuster les options de lecture si nécessaire
    return df_rgb


# =============================================================================
# 2. Calcul des valeurs colorimétriques (XYZ) de l’éclairage
# =============================================================================
def calcul_xyz_eclairage(df_cmf: pd.DataFrame, df_lights: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule pour chaque éclairage (chaque colonne de df_lights, hormis la colonne de longueur d’onde)
    les valeurs X(w), Y(w), Z(w) intégrées sur la plage spectrale, indépendamment de tout objet.
    Ici, on calcule en fait le 'blanc' (XYZ_illuminant) pour chaque éclairage,
    utile ensuite comme référence pour le L*a*b*.

    Retourne un DataFrame indexé par le nom d’éclairage (par ex. Eclairage_1, Eclairage_2)
    contenant [X_ill, Y_ill, Z_ill].
    """
    # Suppose que la 1ère colonne de df_lights est la longueur d’onde identique à df_cmf
    # et que les colonnes suivantes sont les intensités spectrales de chaque éclairage.
    wave_col = df_lights.columns[0]

    xyz_illuminants = {}
    for light_col in df_lights.columns[1:]:
        # Produit spectre éclairage * CMF, puis somme
        X_val = np.trapz(df_lights[light_col] * df_cmf["x_bar"], x=df_lights[wave_col])
        Y_val = np.trapz(df_lights[light_col] * df_cmf["y_bar"], x=df_lights[wave_col])
        Z_val = np.trapz(df_lights[light_col] * df_cmf["z_bar"], x=df_lights[wave_col])

        xyz_illuminants[light_col] = [X_val, Y_val, Z_val]

    df_xyz_ill = pd.DataFrame.from_dict(
        xyz_illuminants,
        orient='index',
        columns=['X_ill', 'Y_ill', 'Z_ill']
    )
    return df_xyz_ill


# =============================================================================
# 3. Calcul des valeurs colorimétriques XYZ et Lab d’un batch sous un éclairage
# =============================================================================
def calcul_xyz_batch_under_light(
        df_batch: pd.DataFrame,
        df_lights: pd.DataFrame,
        df_cmf: pd.DataFrame,
        xyz_ill: np.array,
        nom_eclairage: str
) -> np.array:
    """
    Calcule les coordonnées XYZ d’un batch donné (spectre d’un patch)
    sous un certain éclairage (nom_eclairage) en intégrant
    la réflectance * spectre de l’éclairage * CMF sur la plage spectrale.

    xyz_ill : les valeurs [X_ill, Y_ill, Z_ill] de l’éclairage (pour normaliser si nécessaire).

    Retourne le triplet XYZ normalisé ou non, selon les besoins.
    """
    # Ici, on suppose que df_batch contient : [Longueur d’onde, Reflectance_batch, ...]
    wave_col = df_batch.columns[0]

    # On multiplie la réflectance par l’éclairage et la CMF
    X_val = np.trapz(df_batch["Reflectance"] * df_lights[nom_eclairage] * df_cmf["x_bar"], x=df_batch[wave_col])
    Y_val = np.trapz(df_batch["Reflectance"] * df_lights[nom_eclairage] * df_cmf["y_bar"], x=df_batch[wave_col])
    Z_val = np.trapz(df_batch["Reflectance"] * df_lights[nom_eclairage] * df_cmf["z_bar"], x=df_batch[wave_col])

    # Normalisation par la valeur Y_ill (on peut aussi normaliser par X_ill+Y_ill+Z_ill, etc. selon la convention)
    X_n = X_val / xyz_ill[0]
    Y_n = Y_val / xyz_ill[1]
    Z_n = Z_val / xyz_ill[2]

    return np.array([X_n, Y_n, Z_n])


def xyz_to_lab(xyz: np.array) -> np.array:
    """
    Convertit un triplet XYZ normalisé dans l’espace CIE L*a*b* (D65 ou autre illuminant de référence).
    Formules simplifiées, à affiner selon vos besoins (FO/T > 0.008856 etc.).
    """

    # On suppose ici la formule standard CIE pour L*a*b*,
    # sans adaptation chromatique : on part d’un XYZ déjà normalisé.

    def f(t):
        delta = 6 / 29
        return t ** (1 / 3) if t > delta ** 3 else (t / (3 * delta ** 2) + 4 / 29)

    X, Y, Z = xyz
    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))
    return np.array([L, a, b])


# =============================================================================
# 4. Recherche de la paire d’éclairages ayant une petite distance pour un batch
#    et une grande pour un autre batch
# =============================================================================
def distance_lab(lab1: np.array, lab2: np.array) -> float:
    """
    Calcule la distance Euclidienne entre deux points dans l'espace L*a*b*.
    """
    return np.linalg.norm(lab1 - lab2)


def trouver_paire_eclairages_critique(
        df_batches: pd.DataFrame,
        df_lights: pd.DataFrame,
        df_cmf: pd.DataFrame,
        df_xyz_ill: pd.DataFrame,
        liste_batchs_interesses: list
):
    """
    Recherche une paire d'éclairages (E1, E2) et une paire de batchs (ex. B1, B2)
    telle que la distance Lab(B1,B2) soit faible sous E1 et forte sous E2, ou vice versa.

    liste_batchs_interesses : liste des noms ou indices des batchs que vous voulez comparer.

    Retourne (eclairage1, eclairage2, batch1, batch2, dist1, dist2)
    ou tout autre objet contenant les informations sur la paire trouvée.
    """
    # Exemple : on parcourt toutes les paires d’éclairages et toutes les paires de batchs,
    # puis on cherche un critère min(distance sous E1) et max(distance sous E2).
    meilleures_valeurs = {
        'E1': None,
        'E2': None,
        'B1': None,
        'B2': None,
        'dist_E1': None,
        'dist_E2': None,
        'ecart': -np.inf
    }

    # Pour chaque paire d’éclairages
    eclairages = df_xyz_ill.index.tolist()  # ex: ["Eclairage_1", "Eclairage_2", ...]
    for i in range(len(eclairages)):
        for j in range(i + 1, len(eclairages)):
            E1, E2 = eclairages[i], eclairages[j]
            xyz_ill_E1 = df_xyz_ill.loc[E1].values
            xyz_ill_E2 = df_xyz_ill.loc[E2].values

            # Pour chaque paire de batchs
            for b1 in liste_batchs_interesses:
                for b2 in liste_batchs_interesses:
                    if b1 >= b2:
                        continue
                    # Récupérer spectre batch b1, batch b2
                    # (Ici on suppose qu’on a un DataFrame par batch séparé,
                    # ou qu’on sait filtrer df_batches selon b1/b2.)
                    df_batch_b1 = df_batches[df_batches["batch_id"] == b1]
                    df_batch_b2 = df_batches[df_batches["batch_id"] == b2]

                    # Calcul Lab sous E1
                    xyz_b1_E1 = calcul_xyz_batch_under_light(df_batch_b1, df_lights, df_cmf, xyz_ill_E1, E1)
                    xyz_b2_E1 = calcul_xyz_batch_under_light(df_batch_b2, df_lights, df_cmf, xyz_ill_E1, E1)
                    lab_b1_E1 = xyz_to_lab(xyz_b1_E1)
                    lab_b2_E1 = xyz_to_lab(xyz_b2_E1)
                    dist_E1 = distance_lab(lab_b1_E1, lab_b2_E1)

                    # Calcul Lab sous E2
                    xyz_b1_E2 = calcul_xyz_batch_under_light(df_batch_b1, df_lights, df_cmf, xyz_ill_E2, E2)
                    xyz_b2_E2 = calcul_xyz_batch_under_light(df_batch_b2, df_lights, df_cmf, xyz_ill_E2, E2)
                    lab_b1_E2 = xyz_to_lab(xyz_b1_E2)
                    lab_b2_E2 = xyz_to_lab(xyz_b2_E2)
                    dist_E2 = distance_lab(lab_b1_E2, lab_b2_E2)

                    # Critère : on veut dist_E1 petite ET dist_E2 grande (ou l'inverse)
                    # On peut définir un critère d’optimisation : par ex. (dist_E2 - dist_E1).
                    # Plus c’est grand, mieux c’est (dist_E2 >> dist_E1).
                    ecart = abs(dist_E2 - dist_E1)
                    if ecart > meilleures_valeurs['ecart']:
                        meilleures_valeurs.update({
                            'E1': E1, 'E2': E2,
                            'B1': b1, 'B2': b2,
                            'dist_E1': dist_E1,
                            'dist_E2': dist_E2,
                            'ecart': ecart
                        })

    return meilleures_valeurs


# =============================================================================
# 5. Extraction des couleurs en RGB
# =============================================================================
def lab_to_rgb(lab: np.array, df_rgb: pd.DataFrame) -> tuple:
    """
    Convertit un L*a*b* en RGB.
    Ici, vous pouvez implémenter une vraie conversion (Lab -> XYZ -> RGB)
    ou utiliser un mapping pré-calculé fourni dans df_rgb.

    Retourne un tuple (R, G, B).
    """
    # Ici, en simplifié, on se contente de forcer des valeurs dans [0,255].
    # Dans la vraie vie, il faut effectuer la matrice de conversion adaptée,
    # puis cliper les valeurs.
    L, a, b = lab
    # EXEMPLE d'approche naïve (à remplacer par un vrai modèle):
    R = np.clip(2 * L + a, 0, 255)
    G = np.clip(2 * L - b, 0, 255)
    B = np.clip(2 * L, 0, 255)
    return (int(R), int(G), int(B))


# =============================================================================
# 6. Génération et enregistrement de l’image au format PNG
# =============================================================================
def generer_image_A5(
        couleur_texte_rgb: tuple,
        couleur_fond_rgb: tuple,
        texte: str = "PINTO_LUTTMANN",
        nom_fichier: str = "resultat.png"
) -> None:
    """
    Génère une image au ratio A5 (par ex. 1748x2480 px ~ 300 DPI)
    et inscrit le texte au centre avec la couleur demandée,
    sur un fond de l’autre couleur.
    Enregistre l’image au format PNG.
    """
    # Ratio A5 = 14.8 cm x 21 cm, ~ 2:3
    largeur = 1748
    hauteur = 2480

    img = Image.new("RGB", (largeur, hauteur), color=couleur_fond_rgb)
    draw = ImageDraw.Draw(img)

    # Choix d'une police (chemin à adapter)
    # font = ImageFont.truetype("arial.ttf", 200)
    # ou police par défaut si on n’en a pas
    font = ImageFont.load_default()

    # Dimensions du texte
    w_text, h_text = draw.textsize(texte, font=font)

    # Position centrée
    x_pos = (largeur - w_text) // 2
    y_pos = (hauteur - h_text) // 2

    draw.text((x_pos, y_pos), texte, fill=couleur_texte_rgb, font=font)
    img.save(nom_fichier, format="PNG")
    print(f"Image enregistrée : {nom_fichier}")


# =============================================================================
# 7. Exemple d’utilisation (main)
# =============================================================================
def main():
    # --- 7.1 Lecture des données ---
    path_cmf = "CMF_xyz_2deg.xlsx"
    path_lights = "Lights_Telelumen.xlsx"
    path_rgb = "RGB.xlsx"
    path_batchs = "spectres_patchs.xlsx"

    df_cmf = lire_data_cmf(path_cmf)
    df_lights = lire_data_eclairage(path_lights)
    df_rgb = lire_data_rgb(path_rgb)
    df_batchs = lire_data_batch(path_batchs)

    # --- 7.2 Calcul XYZ pour chaque éclairage ---
    df_xyz_ill = calcul_xyz_eclairage(df_cmf, df_lights)

    # --- 7.3 Recherche de la paire d'éclairages et de batchs critiques ---
    # Supposez qu’on a une liste des batchs (patchs) qu’on veut comparer.
    liste_batchs_interesses = [1, 2, 3, 4]  # Exemples d’ID ou indices

    result = trouver_paire_eclairages_critique(
        df_batchs, df_lights, df_cmf, df_xyz_ill, liste_batchs_interesses
    )

    E1, E2 = result['E1'], result['E2']
    B1, B2 = result['B1'], result['B2']
    dist_E1, dist_E2 = result['dist_E1'], result['dist_E2']

    print("Meilleure paire d'éclairages trouvée :", E1, "et", E2)
    print("Meilleure paire de batchs :", B1, "et", B2)
    print("Distance Lab sous E1 :", dist_E1)
    print("Distance Lab sous E2 :", dist_E2)

    # --- 7.4 Exemple d’extraction de couleurs pour l’image ---
    # Recalcul Lab du batch B1 sous E1 (par exemple)
    xyz_ill_E1 = df_xyz_ill.loc[E1].values
    df_batch_b1 = df_batchs[df_batchs["batch_id"] == B1]
    xyz_b1_E1 = calcul_xyz_batch_under_light(df_batch_b1, df_lights, df_cmf, xyz_ill_E1, E1)
    lab_b1_E1 = xyz_to_lab(xyz_b1_E1)
    rgb_b1_E1 = lab_to_rgb(lab_b1_E1, df_rgb)

    # Recalcul Lab du batch B2 sous E1 (ou E2, selon vos besoins)
    df_batch_b2 = df_batchs[df_batchs["batch_id"] == B2]
    xyz_b2_E2 = calcul_xyz_batch_under_light(df_batch_b2, df_lights, df_cmf, df_xyz_ill.loc[E2].values, E2)
    lab_b2_E2 = xyz_to_lab(xyz_b2_E2)
    rgb_b2_E2 = lab_to_rgb(lab_b2_E2, df_rgb)

    # --- 7.5 Générer l’image (texte couleur batch B1_E1, fond couleur batch B2_E2) ---
    generer_image_A5(rgb_b1_E1, rgb_b2_E2, texte="PINTO_LUTTMANN", nom_fichier="resultat.png")


# Point d’entrée standard (si on veut exécuter ce script directement)
if __name__ == "__main__":
    main()
