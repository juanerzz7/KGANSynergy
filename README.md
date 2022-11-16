# KGANSynergy
This is the implementation code of KGANSynergy, a novel end-to-end Knowledge Graph Attention Network for Drug Synergy (KGANSynergy), which utilizes neighbor information of known drugs/cell lines effectively. KGANSynergy uses knowledge graph (KG) hierarchical propagation to find multi-source neighbor nodes for drugs and cell lines. The knowledge graph attention network is designed to distinguish the importance of neighbors in a KG through a multi-attention mechanism and then aggregate the entity’s neighbor node information to enrich the entity. Finally, the learned drug and cell line embeddings can be utilized to predict the synergy of drug combinations.

## Environment Requirement
Python == 3.6
pandas == 1.1.5
numpy == 1.14.0
pytorch == 1.6.0
h5py == 3.1.0
scipy == 1.5.3

## Dataset
* DrugCombDB(http://drugcombdb.denglab.org/main) is a database with the largest number of drug combinations to date.
* Oncology-Screen(http://www.bioinf.jku.at/software/DeepSynergy/) is an unbiased oncology compound screen datasets.
* Protein-Protein Interaction Network is a comprehensive human interactome network.
* Drug-protein Associations are based on FDA-approved or clinically investigational drugs.
* Cell-protein Associations is harvested from the Cancer Cell Line Encyclopedia.
