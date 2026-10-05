"""mvdecoder: a standalone two-stage movement decoder for stimulated HD-EMG.

The two stages are a move/rest gate and a gesture head, both trained under stimulation.
An optional third head, the clean head, takes the windows where stimulation is off. Fit
it and StreamingDecoder routes every window to the head that owns its regime.

Public API:

  Config      DecoderConfig, EXPERLANGEN, PREERLPLOTS_DAI5
  Model       TwoStageDecoder
  Streaming   StreamingDecoder
  Control     Decider
  Stim source StimSource, TriggerStimSource, PulseTimesStimSource, MaskStimSource
  IO          load_recording, load_feat_cache, load_good_indices
  Offline     extract_offline, decide_offline

Metrics live in mvdecoder.eval. The core (everything except TwoStageDecoder.fit) is
numpy/scipy only; sklearn is used only when fitting.
"""
from .config import DecoderConfig, EXPERLANGEN, PREERLPLOTS_DAI5, EXP13072026
from .decoder import TwoStageDecoder
from .streaming import StreamingDecoder
from .decider import Decider
from .stim_source import (StimSource, TriggerStimSource, PulseTimesStimSource,
                          MaskStimSource)
from .io import load_recording, load_feat_cache, load_good_indices
from .offline import extract_offline, decide_offline

__all__ = [
    "DecoderConfig", "EXPERLANGEN", "PREERLPLOTS_DAI5", "EXP13072026",
    "TwoStageDecoder", "StreamingDecoder", "Decider",
    "StimSource", "TriggerStimSource", "PulseTimesStimSource", "MaskStimSource",
    "load_recording", "load_feat_cache", "load_good_indices",
    "extract_offline", "decide_offline",
]
