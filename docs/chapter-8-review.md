# Chapter 8 alignment and adversarial review

The explicit `native_pixels` profile matches the thesis's main native-target experiment at the mathematical and software levels. The default `emulated_cells` profile preserves the recovered implementation and has a different spatial step. This review does not certify physical operation of the refactored scanner or reproduce the HY300's optical behavior.

## Method alignment

The thesis locators below refer to Chapter 8's displacement, scaling and fusion discussion and its section “Resultados usando REOSSI para crear una nube de puntos”.

| Thesis operation | Maintained implementation | Assessment |
| --- | --- | --- |
| Vertical patterns move right; horizontal patterns move down in the projector image | `shift_native_patterns` in `structured_light/patterns.py` | Enlarges first, shifts the corresponding bit frames without wrapping, preserves references and encodes zero in the exposed prefix. |
| Native 1920→emulated 320: `F_e=6`, `N_d=5`, `step=W_HD=1` | `ScanPlan` and `pipeline.generate_pattern_sequence` with `native_pixels` | Six positions, 0 through 5, with one projector-raster pixel between positions. The extra-shift count is distinct from total positions. |
| `s_d=F_e*(m_d-1)+d*step+1` (equation 8.18) | `scale_identifiers` in `reconstruction/identifiers.py` | Retains the one-based transform, dtype and source sentinel policy. |
| Maximum of scaled displacement layers at each camera pixel | `fuse_identifier_maps` and `reconstruct_identifier_maps` | Preserves column/row alignment and historical stack/reduction axes. |
| Repeated captures before integer decoding | `aggregate_repeated_bits` in `structured_light/decoding.py` | Separate from REOSSI displacement fusion; this stage belongs to the inherited scanner foundation. |
| Decode, then reconstruct 3D using calibrated geometry | Shared decoder, physical/virtual acquisition and reconstruction workflow | Common processing path. New physical frame association and metric accuracy remain untested on the rig. |

`W_LD` and `W_HD` describe fringe widths in projector-raster pixels. They are not image dimensions. A pixel in the configured input raster is also not automatically a physical panel pixel; display scaling and the projector's optical mapping matter.

## Findings and dispositions

1. **Corrected: placement could be lost when reusing a recording.** Capture initially stored placement only in `acquisition.json`; reusing its numerical `scan.json` could select the default cell profile. Initial and completed capture metadata now retain placement and effective raster in the reusable file. Scene outputs and replay preserve explicitly supplied placement too. A capture→replay→pattern-export test checks exact regenerated arrays. Unknown historical placement is not inferred.

2. **Corrected: an active encoding warning was classified as historical.** Native plan inspection initially moved every note into `source_compatibility_notes`. The power-of-two terminal-identifier limitation also affects native placement. Active warnings now remain under `compatibility_notes`; only the whole-cell note is separated for the native profile.

3. **Corrected: parts of the method documentation were stale.** The method, architecture and development notes now distinguish both profiles and identify the explicit choice required for the main thesis experiment. The physical guide describes the same selection and its limits.

4. **Retained limitation: one-based power-of-two grids.** The inherited `ceil(log2(max_size))` bit count cannot represent the final one-based identifier when the maximum emulated dimension is a power of two. For example, ideal 8×6→24×18 captures with factor 3 reconstruct the last three columns as `[21,21,21]` instead of `[22,23,24]`. This is an encoding limitation, not a displacement error. The main 320×180 grid is unaffected. Changing the bit count would change sequence length and recording compatibility, so this integration preserves it with an active warning.

5. **Explicit choice: the default remains source-compatible.** `emulated_cells` shifts complete low-resolution cells before enlargement. It does not implement the main thesis experiment's one-pixel step. Select `native_pixels` explicitly in the CLI or configuration for that experiment; no default is silently changed.

6. **Unresolved scope: intermediate target resolution.** The chapter's general `step=W_LD/W_HD` expression conflicts with its explicit native-target step and its intermediate example's spatial/identifier units. The retained intermediate path is not certified as physically equivalent. The new native profile rejects intermediate targets and unequal/noninteger enlargement factors rather than guessing another transform.

7. **Unexecuted: physical validation of the new port.** Automated capture tests use mocked devices. Ideal digital recovery does not demonstrate synchronization, exposure, projected step, blur tolerance or metric accuracy on the HY300/OAK-1. This does not negate the original scanner's successful physical use. Follow the [physical validation workflow](physical-validation.md) when the rig is available.

## Checks and interpretation

The local public suite passes 297 tests on Python 3.12, including native edge movement, unchanged references, complementary pairs, zero prefixes, early budget rejection, default-profile continuity, reusable configuration and replay. Ruff checks and formatting pass. An external Matplotlib/PyParsing deprecation warning remains.

Independent tiny ideal-capture checks also recover every native identifier for 9×6→54×36 and 10×6→60×36 at factor 6, and 12×8→48×32 at factor 4. These checks use float64 images and the shared classifier/decoder/fusion, independently of the triangle renderer. The power-of-two counterexample above was retained when assessing generality.

The [three-mode comparison](comparison.md) keeps the rig, scene and sensor fixed and reports coverage, identifier error, 3D error and capture cost separately. Its ideal conventional result can equal REOSSI. It does not establish that REOSSI outperforms an ideal conventional system or quantify the thesis's physical improvement.
