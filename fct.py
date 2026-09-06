import pandas as pd


def compare_covenants(df_excel, df_json):
    """
    Compare les covenants entre df_excel et df_json.

    Colonnes attendues dans les deux DataFrames :
        - filename
        - frequency
        - covenant_name
        - covenant_category

    Matching :
        1. même filename
        2. même covenant_category
        3. covenant_name identique OU l'un contenu dans l'autre

    Si les deux covenant_name matchent :
        - on garde le nom le plus long.

    Sortie :
        - filename
        - covenant_name
        - covenant_category
        - exist_json
        - exist_excel
        - frequency_json
        - frequency_excel
        - diff_frequency
    """

    # ---------------------------------------------------------
    # 1. Vérification des colonnes
    # ---------------------------------------------------------

    required_columns = {
        "filename",
        "frequency",
        "covenant_name",
        "covenant_category",
    }

    for df, name in [
        (df_excel, "df_excel"),
        (df_json, "df_json"),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} ne contient pas les colonnes : {missing}"
            )

    # On travaille sur des copies
    excel = df_excel.copy()
    json_df = df_json.copy()

    # ---------------------------------------------------------
    # 2. Fonction de normalisation
    # ---------------------------------------------------------

    def normalize(value):
        """
        Normalisation utilisée uniquement pour les comparaisons.
        """
        if pd.isna(value):
            return ""

        return " ".join(
            str(value)
            .strip()
            .lower()
            .split()
        )

    # ---------------------------------------------------------
    # 3. Colonnes normalisées
    # ---------------------------------------------------------

    for df in [excel, json_df]:

        df["_filename_norm"] = (
            df["filename"]
            .apply(normalize)
        )

        df["_category_norm"] = (
            df["covenant_category"]
            .apply(normalize)
        )

        df["_name_norm"] = (
            df["covenant_name"]
            .apply(normalize)
        )

        df["_frequency_norm"] = (
            df["frequency"]
            .apply(normalize)
        )

    # ---------------------------------------------------------
    # 4. Fonction de matching des covenant_name
    # ---------------------------------------------------------

    def covenant_names_match(name_excel, name_json):
        """
        Retourne True si :
        - les noms sont identiques
        - ou l'un est contenu dans l'autre
        """

        if not name_excel or not name_json:
            return False

        return (
            name_excel == name_json
            or name_excel in name_json
            or name_json in name_excel
        )

    # ---------------------------------------------------------
    # 5. Stockage des résultats
    # ---------------------------------------------------------

    results = []

    # Permet de savoir quelles lignes JSON ont déjà été associées
    matched_json_indices = set()

    # ---------------------------------------------------------
    # 6. Parcours des covenants Excel
    # ---------------------------------------------------------

    for excel_idx, excel_row in excel.iterrows():

        # On cherche d'abord seulement dans :
        # - le même fichier
        # - la même catégorie

        candidates = json_df[
            (json_df["_filename_norm"] == excel_row["_filename_norm"])
            &
            (json_df["_category_norm"] == excel_row["_category_norm"])
            &
            (~json_df.index.isin(matched_json_indices))
        ]

        matches = []

        # -----------------------------------------------------
        # Chercher les covenant_name compatibles
        # -----------------------------------------------------

        for json_idx, json_row in candidates.iterrows():

            if covenant_names_match(
                excel_row["_name_norm"],
                json_row["_name_norm"],
            ):

                # Score permettant de privilégier :
                # 1. exact match
                # 2. sinon le match le plus proche

                exact_match = (
                    excel_row["_name_norm"]
                    == json_row["_name_norm"]
                )

                length_difference = abs(
                    len(excel_row["_name_norm"])
                    - len(json_row["_name_norm"])
                )

                score = (
                    1 if exact_match else 0,
                    -length_difference,
                )

                matches.append(
                    (
                        score,
                        json_idx,
                        json_row,
                    )
                )

        # -----------------------------------------------------
        # Cas 1 : covenant trouvé dans JSON
        # -----------------------------------------------------

        if matches:

            # Le meilleur match :
            # exact match prioritaire,
            # sinon différence de longueur minimale
            matches.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            _, json_idx, json_row = matches[0]

            matched_json_indices.add(json_idx)

            # -------------------------------------------------
            # Garder le covenant_name le plus long
            # -------------------------------------------------

            excel_name = (
                ""
                if pd.isna(excel_row["covenant_name"])
                else str(excel_row["covenant_name"]).strip()
            )

            json_name = (
                ""
                if pd.isna(json_row["covenant_name"])
                else str(json_row["covenant_name"]).strip()
            )

            if len(json_name) > len(excel_name):
                final_name = json_name
            else:
                final_name = excel_name

            # -------------------------------------------------
            # Comparaison des fréquences
            # -------------------------------------------------

            diff_frequency = int(
                excel_row["_frequency_norm"]
                != json_row["_frequency_norm"]
            )

            results.append(
                {
                    "filename": excel_row["filename"],
                    "covenant_name": final_name,
                    "covenant_category": excel_row["covenant_category"],

                    "exist_json": 1,
                    "exist_excel": 1,

                    "frequency_json": json_row["frequency"],
                    "frequency_excel": excel_row["frequency"],

                    "diff_frequency": diff_frequency,
                }
            )

        # -----------------------------------------------------
        # Cas 2 : covenant uniquement dans Excel
        # -----------------------------------------------------

        else:

            results.append(
                {
                    "filename": excel_row["filename"],
                    "covenant_name": excel_row["covenant_name"],
                    "covenant_category": excel_row["covenant_category"],

                    "exist_json": 0,
                    "exist_excel": 1,

                    "frequency_json": pd.NA,
                    "frequency_excel": excel_row["frequency"],

                    "diff_frequency": 0,
                }
            )

    # ---------------------------------------------------------
    # 7. Ajouter les covenants présents uniquement dans JSON
    # ---------------------------------------------------------

    for json_idx, json_row in json_df.iterrows():

        if json_idx not in matched_json_indices:

            results.append(
                {
                    "filename": json_row["filename"],
                    "covenant_name": json_row["covenant_name"],
                    "covenant_category": json_row["covenant_category"],

                    "exist_json": 1,
                    "exist_excel": 0,

                    "frequency_json": json_row["frequency"],
                    "frequency_excel": pd.NA,

                    "diff_frequency": 0,
                }
            )

    # ---------------------------------------------------------
    # 8. Création du DataFrame final
    # ---------------------------------------------------------

    df_result = pd.DataFrame(results)

    return df_result



df_comparison = compare_covenants(
    df_excel=df_excel,
    df_json=df_json,
)

df_comparison