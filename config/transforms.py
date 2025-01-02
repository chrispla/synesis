from torch_audiomentations import AddColoredNoise, Gain, PitchShift

configs = {
    "AddColoredNoise": {
        "class": AddColoredNoise,
        "params": {
            "color": "white",
            "min_snr_in_db": 10,
            "max_snr_in_db": 20,
            "p": 1,
            "mode": "per_example",
        },
        "step": 1,
    },
    "Gain": {
        "class": Gain,
        "params": {
            "min_gain_in_db": -10,
            "max_gain_in_db": 10,
            "p": 1,
            "mode": "per_example",
        },
        "step": 1,
    },
    "PitchShift": {
        "class": PitchShift,
        "params": {
            "min_semitones": -6,
            "max_semitones": 6,
            "p": 1,
            "mode": "per_example",
        },
        "step": 1,
    },
}
