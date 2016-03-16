# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 11:12:19 2016

@author: tanfan.zjh
"""

DNA = ['Gene','Gene_Family','Box','Promoter']
Aminoacid_Sequence = ['Protein','Protein_Family','Protein_Complex','Protein_Domain']
DNA_Product = ['RNA',Aminoacid_Sequence]
Molecule = ['Hormone',DNA,DNA_Product]
Dynamic_Process = ['Regulatory_Network','Metabolic pathway']
Biological_context = ['Genotype','Tissue','Development_Phase']
Context = [Biological_context,'Environmental_Factor']

entities = [Molecule,Dynamic_Process,Context]
all_entities = ['Gene','Gene_Family','Box','Promoter','Protein','Protein_Family',
                'Protein_Complex','Protein_Domain','RNA','Hormone',
                'Regulatory_Network','Metabolic pathway','Genotype','Tissue',
                'Development_Phase','Environmental_Factor']
####### when and where ##########
## A Process occurs in a given Genotype. 
Occurrence_In_Genotype = {'Process': ['Regulatory_Network', 'Pathway'],
                          'Genotype': ['Genotype'] } 
# A Molecule or Element is present in a given Genotype. 
Exists_In_Genotype = {'Molecular':['Gene', 'Gene_Family', 'Box', 'Promoter', 'RNA', 
                         'Protein', 'Protein_Family', 'Protein_Complex', 
                         'Protein_Domain', 'Hormone'],
                      'Element':['Tissue', 'Development_Phase', 'Genotype'],
                      'Genotype': ['Genotype']}##Molecule et Element are mutually exclusive
# A Molecule is present during a given Developmental phase.
Exists_At_Stage = {'Funtional_molecular':['RNA','Protein','Protein_Family',
                                   'Protein_Complex','Protein_Domain', 'Hormone'],
                    'Development':['Development_Phase']}
##  A Process occurs during a given Developmental Phase. 
Occurs_During = {'Process': ['Regulatory_Network', 'Pathway'],
                     'Development': ['Development_Phase']}
## 分子在组织上
Is_Localized_In = {'Funtional_molecular':['RNA','Protein','Protein_Family',
                                   'Protein_Complex','Protein_Domain', 'Hormone'],
                   'Dynamic_Process':['Regulatory_Network', 'Pathway'],
                   'Target_Tissue': ['Tissue']}##(Functional_Molecule et Process are mutually exclusive)

#############################################################################################
##### Function #######
## A Molecule is involved in a Dynamic Process. 
Is_Involved_In_Process = {'Molecule':['Gene', 'Gene_Family', 'Box', 'Promoter', 'RNA', 
                         'Protein', 'Protein_Family', 'Protein_Complex', 'Protein_Domain',
                         'Hormone'],
                         'Dynamic_Process':['Regulatory_Network', 'Pathway']}
## Gene encodes protein or RNA encodes Protein or DNA encodes RNA                             
Transcribes_Or_Translates_To = {'Source':['Gene', 'Gene_Family', 'RNA'],
                                'DNA_Product':['RNA', 'Protein', 'Protein_Family',
                                               'Protein_Complex', 'Protein_Domain']}
Is_Functionally_Equivalent_To = {'Element1':'All Entities','Element2':'All Entities'}

#############################################################################################
## 调控事件 ##
##A Molecule, Dynamic Process or Context regulates the accumulation of a Functional Molecule 
Regulates_Accumulation = {'Agent':'All Entities',
                          'Funtional_molecule':['RNA','Protein','Protein_Family',
                                   'Protein_Complex','Protein_Domain', 'Hormone']}
Regulates_Expression = {'Agent':'All Entities',
                        'DNA':['Gene', 'Gene_Family', 'Box', 'Promoter']}
##  A Molecule, Dynamic Process or Context regulates the activity of a Development phase.                                  
Regulates_Development_Phase = {'Agent':'All Entities',
                               'Development': ['Development_Phase']}
Regulates_Molecule_Activity = {'Agent':'All Entities',
                               'Molecule':['Protein', 'Protein_Family', 
                                            'Protein_Complex', 'Hormone']}#Molecule: Amino acid sequence| Hormone  (Protein, Protein_Family, Protein_Complex, Hormone )
Regulates_Process = {'Agent':'All Entities',
                     'Dynamic_Process':['Regulatory_Network', 'Pathway']}
                 
Regulates_Tissue_Development = {'Agent':'All Entities',
                                'Target_Tissue': ['Tissue']}
                                
#############################################################################################
## Composition_and_Membership ##
#核苷酸序列在DNA
Composes_Primary_Structure = {'DNA_Part':['Box','Promoter'],
                              'DNA':['Gene', 'Gene_Family', 'Box', 'Promoter']}

# 氨基酸序列在蛋白质复合物
Composes_Protein_Complex = {'Amino_Acid_Sequence':['Protein','Protein_Family',
                                                   'Protein_Complex','Protein_Domain'],
                            'Protein_Complex':['Protein_Complex']}
## This relation is to be used between entities of the same nature
Is_Member_Of_Family = {'Element': ['Gene', 'Gene_Family', 'RNA', 'Protein', 
                                   'Protein_Family', 'Protein_Domain'],
                       'Family': ['Gene_Family', 'RNA', 'Protein_Family']}

# 蛋白质域是蛋白质的一部分                       
Is_Protein_Domain_Of = {'Domain': ['Protein_Domain'],
                        'Product': ['RNA', 'Protein', 'Protein_Family', 'Protein_Complex', 
                                    'Protein_Domain']}                     
Has_Sequence_Identical_To = {'Element1':'All Entities','Element2':'All Entities'}

#############################################################################################
## Interaction ##
## DNA-DNA interaction
Interacts_With = {'Agent':['Gene', 'Gene_Family', 'Box', 'Promoter','Gene', 'Gene_Family', 
                           'Box', 'Promoter', 'Protein', 'Protein_Family', 'Protein_Complex', 
                           'Protein_Domain'],#DNA | Amino acid sequence
                  'Target':['Gene', 'Gene_Family', 'Box', 'Promoter','Gene', 'Gene_Family', 
                           'Box', 'Promoter', 'Protein', 'Protein_Family', 'Protein_Complex', 
                           'Protein_Domain']}#DNA | Amino acid sequence
Binds_To = {'Funtional_molecule':['RNA','Protein','Protein_Family',
                                   'Protein_Complex','Protein_Domain', 'Hormone'],
            'Molecule':['Gene', 'Gene_Family', 'Box', 'Promoter', 'RNA', 
                         'Protein', 'Protein_Family', 'Protein_Complex', 
                         'Protein_Domain', 'Hormone']}
#############################################################################################                                               
Is_Linked_To = {'Agent':'all',
                'Target':'all'}

Where_and_When = [Occurrence_In_Genotype,Exists_In_Genotype,Exists_At_Stage,
                  Occurs_During,Is_Localized_In]
Function = [Is_Involved_In_Process,Transcribes_Or_Translates_To,Is_Functionally_Equivalent_To]
Regulation = [Regulates_Accumulation,Regulates_Expression,Regulates_Development_Phase,
              Regulates_Molecule_Activity,Regulates_Process,Regulates_Tissue_Development]
Composition_and_Membership = [Composes_Primary_Structure,Composes_Protein_Complex,
                              Is_Protein_Domain_Of,Is_Member_Of_Family,
                              Has_Sequence_Identical_To]
Interaction = [Binds_To,Interacts_With]

ALL_EVENT = [Where_and_When,Function,Regulation,Composition_and_Membership,Interaction]

RELATIONS = ['Is_Member_Of_Family','Is_Linked_To','Regulates_Tissue_Development',
             'Has_Sequence_Identical_To','Is_Protein_Domain_Of','Regulates_Process',
             'Composes_Protein_Complex','Occurs_During','Interacts_With','Is_Localized_In',
             'Composes_Primary_Structure','Occurs_In_Genotype','Binds_To','Exists_In_Genotype',
             'Regulates_Accumulation','Transcribes_Or_Translates_To','Regulates_Development_Phase',
             'Regulates_Expression','Regulates_Molecule_Activity','Is_Functionally_Equivalent_To',
             'Exists_At_Stage','Is_Involved_In_Process']



