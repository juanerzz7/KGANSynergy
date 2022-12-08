# KGANSynergy
This is the implementation code of KGANSynergy, a novel end-to-end Knowledge Graph Attention Network for Drug Synergy (KGANSynergy), which utilizes neighbor information of known drugs/cell lines effectively. KGANSynergy uses knowledge graph (KG) hierarchical propagation to find multi-source neighbor nodes for drugs and cell lines. The knowledge graph attention network is designed to distinguish the importance of neighbors in a KG through a multi-attention mechanism and then aggregate the entity’s neighbor node information to enrich the entity. Finally, the learned drug and cell line embeddings can be utilized to predict the synergy of drug combinations.

[The framework of KGANSynergy](https://github.com/juanerzz7/KGANSynergy/blob/main/fig.pdf)

## Environment Requirement
Python == 3.6<br>
pandas == 1.1.5<br>
numpy == 1.14.0<br>
pytorch == 1.6.0<br>
h5py == 3.1.0<br>
scipy == 1.5.3<br>

## Dataset
### Original data
* [DrugCombDB](http://drugcombdb.denglab.org/main) is a database with the largest number of drug combinations to date.
* [Oncology-Screen](http://www.bioinf.jku.at/software/DeepSynergy/) is an unbiased oncology compound screen datasets.
* [Cell-protein Associations](https://maayanlab.cloud/Harmonizome/dataset/CCLE+Cell+Line+Gene+Expression+Profiles) is harvested from the Cancer Cell Line Encyclopedia.
* Protein-Protein Interaction Network is a comprehensive human interactome network.
* Drug-protein Associations are based on FDA-approved or clinically investigational drugs.

### Processed data
* rating_final.txt --Drug combination data
* kg_final2.txt --Knowledge graph triple

## Running the code
```
python train.py
```

## Other important files
* model.py --KGANSynergy architecture
* load_gat.py -- Read drug combination data and knowledge graph data
* dataloader4KGNN.py --Generate knowledge graph index and adjacency list

## Default parameters of the scripts
Parameter  | DrugCombDB  | Oncology-Screen
 ---- | ----- | ------  
 Batch size  | 1024 | 32 
 Learning rate  | 0.001 | 0.0001   
 Dim  | 128 | 64   
 Layer  | 2 | 2   
 L2 weight  | 1e-4 | 1e-4   
 Optimizer  | Adam | Adam   
 
 
