from pathlib import Path
import pandas as pd


ID_COLUMNS = ["state", "county", "community", "communityname", "fold"]
TARGET_COLUMN = "ViolentCrimesPerPop"


def load_communities_crime(
    data_path: str | Path,
    names_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load the normalized Communities and Crime dataset.

    Parameters
    ----------
    data_path:
        Path to communities.data.
    names_path:
        Optional path to communities.names. Not required if column names
        are provided manually.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with missing values converted to NaN.
    """

    data_path = Path(data_path)

    column_names = get_column_names()

    df = pd.read_csv(
        data_path,
        header=None,
        names=column_names,
        na_values="?",
    )

    return df


def get_column_names() -> list[str]:
    """
    Column names for the normalized UCI Communities and Crime dataset.
    """

    return [
        "state", "county", "community", "communityname", "fold",
        "population", "householdsize", "racepctblack", "racePctWhite",
        "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29",
        "agePct16t24", "agePct65up", "numbUrban", "pctUrban",
        "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc",
        "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc",
        "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap",
        "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov",
        "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad",
        "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu",
        "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf",
        "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv",
        "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
        "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids",
        "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig",
        "PctImmigRecent", "PctImmigRec5", "PctImmigRec8",
        "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
        "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly",
        "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup",
        "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous",
        "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR",
        "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc",
        "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt",
        "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart",
        "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian",
        "RentHighQ", "MedRent", "MedRentPctHousInc",
        "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters",
        "NumStreet", "PctForeignBorn", "PctBornSameState",
        "PctSameHouse85", "PctSameCity85", "PctSameState85",
        "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps",
        "LemasSwFTFieldPerPop", "LemasTotalReq",
        "LemasTotReqPerPop", "PolicReqPerOffic",
        "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite",
        "PctPolicBlack", "PctPolicHisp", "PctPolicAsian",
        "PctPolicMinor", "OfficAssgnDrugUnits",
        "NumKindsDrugsSeiz", "PolicAveOTWorked",
        "LandArea", "PopDens", "PctUsePubTrans",
        "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr",
        "LemasGangUnitDeploy", "LemasPctOfficDrugUn",
        "PolicBudgPerPop", "ViolentCrimesPerPop",
    ]


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate predictors and target.
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y