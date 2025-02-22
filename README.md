# MouseEmbryoSimulator
This repository contains code to simulate the first mammalian cell fate decision when trophectoderm (TE) cells segregate from inner cell mass (ICM) cells in the early mouse embryo. 

Here, we chose CDX2 as a marker of TE and another transcription factor SOX2 as a marker of ICM and study their dynamics together with an upstream transcriptional regulator YAP. We collected live imaging data for embryos between 8- and 32-cell stages, relevant for this cell fate decision. Our dataset includes pairwise YAP-CDX2 and YAP-SOX2 observations which we use to model the regulation of the key fate determinants by YAP. We use the pairwise data to model dynamics of every cell in an embryo in the YAP-SOX2-CDX2 phase space via dynamic Bayesian networks (DBNs). The models are used to provide posterior distributions on relevant variables or simulate dynamics of an embryo between 8 and 32 cell stages taking the lineage relationships into account. 

Here is the structure of this repository.

|-- **data**: preprocessed live imaging data for YAP-CDX2 and YAP-SOX2 that is used to run the example notebooks \
|&emsp; |-- **Yap_Cdx2**: data for YAP-CDX2 \
|&emsp; |-- **Yap_Sox2**: data for YAP-SOX2 \
|-- **Sox2_for_YSC.ipynb**: Jupyter notebook demonstrating dynamic Bayesian network analysis on YAP-SOX2 data; should be run first \
|-- **Cdx2_for_YSC.ipynb**: Jupyter notebook demonstrating dynamic Bayesian network analysis on YAP-CDX2 data and data fusion to obtain joint YAP-SOX2-CDX2; should be run after  \
|-- **Align_time.py**: preprocessing utility \
|-- **Norm_TF.py**: preprocessing utility \
|-- **Analysis_YSC.py**: auxiliary methods for DBN analysis \
|-- **Sim_YSC.py**: embryonic dynamics simulator \
|-- **Visual_YSC.py**: visualization methods \


