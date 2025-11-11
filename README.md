# rootlab
Toolkit for converting ML-based root tracings into quantitative root architecture data.


The **RootLab** package provides a complete workflow for analyzing **root system architecture** from traced image data. It is designed to work seamlessly with the ML-based tracing outputs of Rhizomatic by Oceanit, converting raw pixel-level annotations into accurate geometric and hierarchical representations of plant root systems.

It reads structured JSON files containing segment-level root traces, links those segments into biological units (branches and systems), and then is able to compute biological order from the traces with a reasonable degree of accuracy. To improve accuracy of the branching architecture and order assignment, I have also developed a method to manually label the taproot / central root axis, and then use that as an anchor on which to base the root architecture.

The overarching goals are to:
- Convert raw traced data into coherent, analyzable structures (branches, systems).
- Quantify and summarize biologically meaningful traits such as root order, length, width, and branching intensity.
- Allow for manual taproot designation to improve order classification
- Visualize and interpret full root systems.

Christian Gray is the author of this project, and developed this tool during his PhD to fulfill gaps in quantitative root analysis. For questions, please contact the author of this project (Christian Gray - czgray@princeton.edu). 
