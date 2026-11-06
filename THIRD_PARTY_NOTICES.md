# Third-party provenance

## Guillaume Charron's scanner

The structured-light foundation is adapted from [3DScanner-GrayCode](https://github.com/guillaume-charron/3DScanner-GrayCode), pinned to commit `6f8abf0257c727eed81cc249edb85d43010c7b3d` (2023-12-22).

Copyright (c) 2023 Guillaume Charron. The complete MIT terms are preserved in [LICENSES/Charron-MIT.txt](LICENSES/Charron-MIT.txt). They apply to the inherited/adapted code, including the Gray-code primitives, photometric classifier, repeated-bit aggregation, calibration, camera adapter, triangulation and visualization foundation.

The modules have been renamed, documented and separated from script-level state. Their names do not imply independent authorship of the inherited algorithms. The local OAK integration, REOSSI displacement orchestration, one-based identifier adaptation, scaling and map fusion come from Rolando Cortez's historical project, with distinctions documented in [the method block map](docs/method.md).

## Sources acknowledged by the reference project

Charron's [original README](https://github.com/guillaume-charron/3DScanner-GrayCode/blob/6f8abf0257c727eed81cc249edb85d43010c7b3d/README.md) and source comments identify additional algorithmic sources. Their attribution is retained here:

- [sjnarmstrong/gray-code-structured-light](https://github.com/sjnarmstrong/gray-code-structured-light): Gray patterns, direct/global illumination and classification algorithm references.
- [caoandong/Projector_Calibration](https://github.com/caoandong/Projector_Calibration/blob/master/visual_calib.py): reference cited beside the triangulation calculation.
- [Gray decoding answer](https://stackoverflow.com/a/72028021): reference cited beside integer Gray decoding.
- [Projector-camera calibration tutorial](https://www.morethantechnical.com/blog/2017/11/17/projector-camera-calibration-the-easy-way/): calibration background cited by Charron.

The reference bibliography includes Xu and Aliaga (2007), *Robust Pixel Classification for 3D Modeling with Structured Light*; Nayar et al. (2006), *Fast Separation of Direct and Global Components of a Scene Using High Frequency Illumination*; Moreno and Taubin (2012), *Simple, Accurate, and Robust Projector-Camera Calibration*; and Yang, Normand and Moreau (2016), *Practical and Precise Projector-Camera Calibration*. Full bibliographic records are available in the linked upstream README.

No source from other structured-light repositories has been merged into this initial project. Original REOSSI contributions do not acquire an MIT license merely because this notice accompanies inherited code. The project's overall licensing decision is pending.
