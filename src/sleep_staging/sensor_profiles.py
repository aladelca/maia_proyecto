"""Public input contracts shared by the API and the research package."""

MAX_EDF_BYTES = 128 * 1024**2
MAX_PARQUET_BYTES = 32 * 1024**2


def channel_profiles() -> dict[str, list[str]]:
    profiles = {}
    for eeg in [["eeg_fpz_cz"], ["eeg_pz_oz"], ["eeg_fpz_cz", "eeg_pz_oz"]]:
        for extra in [[], ["eog"], ["emg"], ["eog", "emg"]]:
            channels = eeg + extra
            profiles["+".join(channels)] = channels
    return profiles


def validate_edf_profile(value: object) -> str:
    if not isinstance(value, str) or (
        value != "auto" and value not in channel_profiles()
    ):
        raise ValueError("La selección debe incluir al menos un EEG compatible.")
    return value


LGBM_PROFILE = "eeg_fpz_cz+eeg_pz_oz+eog+emg"


def validate_lgbm_profile(value: object) -> str:
    if value not in ("auto", LGBM_PROFILE):
        raise ValueError("LightGBM requiere los dos EEG, EOG y EMG submental.")
    return LGBM_PROFILE
