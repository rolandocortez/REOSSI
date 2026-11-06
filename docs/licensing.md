# Licensing and contribution boundaries

Original REOSSI contributions by Rolando Cortez Garcia are licensed under the
[PolyForm Noncommercial License 1.0.0](../LICENSE). The required copyright notice
is in [NOTICE](../NOTICE). This applies to original contributions in the public
REOSSI project, including original additions in files containing adapted code.
It does not license separate research packages or transfer rights to the paper.

Third-party material retains its existing license. Charron-derived portions keep
the permissions and obligations of [MIT](../LICENSES/Charron-MIT.txt), as described
in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The PolyForm terms do not
remove those permissions for the inherited material itself. Dependencies are
separately licensed and are not relicensed by this project.

## Original method and implementation

| Contribution | Implementation | Scope |
| --- | --- | --- |
| Native, emulated and target-grid planning | [`ScanPlan`](../src/reossi/plan.py) | Original configuration and orchestration of recovered REOSSI parameters. |
| Shifted acquisition orchestration | [`generate_pattern_sequence`](../src/reossi/pipeline.py), [`decode_displacement`](../src/reossi/pipeline.py) | Original composition; calls to Gray primitives do not make their inherited implementation original. |
| Per-layer identifier transformation | [`scale_identifiers`](../src/reossi/reconstruction/identifiers.py) | Rolando's recovered REOSSI operation and its maintained implementation. |
| Fusion of transformed displacement maps | [`fuse_identifier_maps`](../src/reossi/reconstruction/identifiers.py), [`reconstruct_identifier_maps`](../src/reossi/pipeline.py) | Rolando's recovered reconstruction operation and pipeline integration. |
| Direct identifier demonstrations | [`simulation/identifiers.py`](../src/reossi/simulation/identifiers.py) | Original identifier constructions and shared transformation/fusion composition. |
| Digital raster and triangle-scene inputs | [`simulation/`](../src/reossi/simulation/) | Project implementation of virtual inputs, geometry, visibility, sensor response and saved truth. Imported Gray and reconstruction primitives retain their own provenance. |
| Maintained application infrastructure | [`cli/`](../src/reossi/cli/), acquisition recording/capture workflows, calibration observation/file workflows, reconstruction workflow, tests and examples | Original application code and additions. This does not change the origin of underlying adapted solvers, adapters or mathematical implementations. |

These contributions are covered by PolyForm Noncommercial. The source map
describes implementation provenance, not exclusive ownership of mathematical
ideas or independently written implementations of the published method.

## Adapted and mixed implementations

| File | Inherited material retained under MIT | Original additions covered by PolyForm Noncommercial |
| --- | --- | --- |
| [`structured_light/patterns.py`](../src/reossi/structured_light/patterns.py) | Conventional Gray-word construction and pattern rasterization. | The local one-based/displaced adaptation and project-specific additions. |
| [`structured_light/illumination.py`](../src/reossi/structured_light/illumination.py) | Baseline direct/global estimation implementation. | Local normalization adaptation and project-specific API additions. |
| [`structured_light/decoding.py`](../src/reossi/structured_light/decoding.py) | Bit classification, maximum over repeated bit classifications, Gray-word decoding and map traversal conventions. | Input/API validation, estimator injection and maintained integration additions. |
| [`calibration/solvers.py`](../src/reossi/calibration/solvers.py) | Adapted ChArUco/projector solver implementation and retained OpenCV call structure. | Configuration, diagnostics, validation and lifecycle-related additions. |
| [`acquisition/opencv.py`](../src/reossi/acquisition/opencv.py) | Adapted threaded OpenCV camera implementation. | Maintained synchronization, validation and cleanup additions. |
| [`reconstruction/geometry.py`](../src/reossi/reconstruction/geometry.py) | Correspondence collection and sine-rule triangulation implementation. | Maintained interface and validation additions. |
| [`visualization.py`](../src/reossi/visualization.py) | Identifier color interpolation and lookup foundation. | Optional plotting interfaces, explicit saving and project-specific views. |

The local OAK integration is an original extension from Rolando's historical
scanner. Its presence does not change the provenance of the shared camera
foundation or of other imported components.

Mixed-file SPDX expressions use `PolyForm-Noncommercial-1.0.0 AND MIT` to record
the licenses applicable to their different contributions. This is not an offer
to choose either license for the entire file. A mixed file as distributed includes
both original and inherited material; the MIT notice alone does not grant MIT
permissions over the original REOSSI additions. See the [method block map](method.md)
for the processing stages and the distinction between repeated-bit aggregation
and final REOSSI displacement fusion.

## Permitted purposes and commercial agreements

PolyForm Noncommercial grants the purposes defined in the full license, including
noncommercial purposes and specified uses by educational, public research and
other listed organizations, regardless of their funding. This is source-available
software with a noncommercial license, not an OSI open-source license.

Uses of the original contributions outside those permissions require a separate
agreement with Rolando Cortez Garcia. PolyForm itself sets no commercial fee and
does not impose a general obligation to notify the author of every permitted use.
This project adds no such condition to the standard license text. The inherited
MIT material remains available under its existing terms independently.
