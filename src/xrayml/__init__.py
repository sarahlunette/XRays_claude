"""xrayml — fracture detection on musculoskeletal radiographs.

The package is split so that the serving path (``xrayml.serving``, ``xrayml.artifacts``,
``xrayml.data.preprocess``, ``xrayml.data.io``) never imports torch: the production image
runs ONNX Runtime only. Everything under ``engine``, ``models``, ``losses`` needs the
``train`` extra.
"""

__version__ = "0.1.0"
