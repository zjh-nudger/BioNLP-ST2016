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

## 分子/过程/上下文
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

ALL_EVENT_CLASS = [Where_and_When,Function,Regulation,Composition_and_Membership,Interaction]



class Event(object):
    def __init__(self,event_class,event_name,e1_name,e1_entity,
                 e2_name,e2_entity,e3_name='',e3_entity=[]):
        self.event_class = event_class
        self.event_name = event_name
        self.e1_name = e1_name
        self.e1_entity = e1_entity
        self.e2_name = e2_name
        self.e2_entity = e2_entity
        self.e3_name = e3_name
        self.e3_entity = e3_entity

def load_event_def():
    event_dir = {}
    event1 = Event(event_class = 'Where_and_When',event_name = 'Occurrence_In_Genotype',
                  e1_name='Process',e1_entity=['Regulatory_Network', 'Pathway'],
                  e2_name='Genotype',e2_entity=['Genotype'])
    event_dir['Occurrence_In_Genotype'] = event1
    event2 = Event(event_class = 'Where_and_When',event_name = 'Exists_In_Genotype',
                  e1_name='Molecular',e1_entity=['Gene', 'Gene_Family', 'Box', 'Promoter', 'RNA', 
                                                 'Protein', 'Protein_Family', 'Protein_Complex', 
                                                 'Protein_Domain', 'Hormone'],
                  e2_name = 'Element',e2_entity=['Tissue', 'Development_Phase', 'Genotype'],
                  e3_name = 'Genotype',e3_entity = ['Genotype'])
    event_dir['Exists_In_Genotype'] = event2
    event3 = Event(event_class = 'Where_and_When',event_name = 'Exists_At_Stage',
                  e1_name='Funtional_molecular',e1_entity=['RNA','Protein','Protein_Family',
                                                           'Protein_Complex','Protein_Domain', 'Hormone'],
                  e2_name='Development',e2_entity=['Development_Phase'])
    event_dir['Exists_At_Stage'] = event3
    event4 = Event(event_class = 'Where_and_When',event_name = 'Occurs_During',
                  e1_name='Process',e1_entity=['Regulatory_Network', 'Pathway'],
                  e2_name='Development',e2_entity=['Development_Phase'])
    event_dir['Occurs_During'] = event4
    ## 分子在组织上
    event5 = Event(event_class = 'Where_and_When',event_name = 'Is_Localized_In',
                  e1_name='Funtional_molecular',e1_entity=['RNA','Protein','Protein_Family',
                                                           'Protein_Complex','Protein_Domain',
                                                           'Hormone'],
                  e2_name = 'Dynamic_Process',e2_entity=['Regulatory_Network', 'Pathway'],
                  e3_name = 'Target_Tissue',e3_entity = ['Tissue'])
    event_dir['Is_Localized_In'] = event5
    ## function
    event6 = Event(event_class = 'Function',event_name = 'Is_Involved_In_Process',
                  e1_name='Molecule',e1_entity=['Gene', 'Gene_Family', 'Box', 'Promoter', 'RNA', 
                                                'Protein', 'Protein_Family', 'Protein_Complex', 'Protein_Domain',
                                                'Hormone'],
                  e2_name='Dynamic_Process',e2_entity=['Regulatory_Network', 'Pathway'])
    event_dir['Is_Involved_In_Process'] = event6
    event7 = Event(event_class = 'Function',event_name = 'Transcribes_Or_Translates_To',
                  e1_name='Source',e1_entity=['Gene', 'Gene_Family', 'RNA'],
                  e2_name='DNA_Product',e2_entity=['RNA', 'Protein', 'Protein_Family',
                                                   'Protein_Complex', 'Protein_Domain'])
    event_dir['Transcribes_Or_Translates_To'] = event7
    event8 = Event(event_class = 'Function',event_name = 'Is_Functionally_Equivalent_To',
                  e1_name='Element1',e1_entity=all_entities,
                  e2_name='Element2',e2_entity=all_entities)
    event_dir['Is_Functionally_Equivalent_To'] = event8
    ## regulate
    event9 = Event(event_class = 'Regulate',event_name = 'Regulates_Accumulation',
                  e1_name='Agent',e1_entity=all_entities,
                  e2_name='Funtional_molecule',e2_entity=['RNA','Protein','Protein_Family',
                                                          'Protein_Complex','Protein_Domain',
                                                          'Hormone'])
    event_dir['Regulates_Accumulation'] = event9
    event10 = Event(event_class = 'Regulate',event_name = 'Regulates_Expression',
                  e1_name='Agent',e1_entity=all_entities,
                  e2_name='DNA',e2_entity=['Gene', 'Gene_Family', 'Box', 'Promoter'])
    event_dir['Regulates_Expression'] = event10
    event11 = Event(event_class = 'Regulate',event_name = 'Regulates_Development_Phase',
                  e1_name='Agent',e1_entity=all_entities,
                  e2_name='Development',e2_entity=['Development_Phase'])
    event_dir['Regulates_Development_Phase'] = event11
    event12 = Event(event_class = 'Regulate',event_name = 'Regulates_Molecule_Activity',
                  e1_name='Agent',e1_entity=all_entities,
                  e2_name='Molecule',e2_entity=['Protein', 'Protein_Family', 
                                                'Protein_Complex', 'Hormone'])
    event_dir['Regulates_Molecule_Activity'] = event12
    event13 = Event(event_class = 'Regulate',event_name = 'Regulates_Process',
                  e1_name='Agent',e1_entity=all_entities,
                  e2_name='Dynamic_Process',e2_entity=['Regulatory_Network', 'Pathway'])
    event_dir['Regulates_Process'] = event13
    event14 = Event(event_class = 'Regulate',event_name = 'Regulates_Tissue_Development',
                  e1_name='Agent',e1_entity=all_entities,
                  e2_name='Target_Tissue',e2_entity=['Tissue'])
    event_dir['Regulates_Tissue_Development'] = event14
    ## compositon
    event15 = Event(event_class = 'Compositon',event_name = 'Composes_Primary_Structure',
                  e1_name='DNA_Part',e1_entity=['Box','Promoter'],
                  e2_name='DNA',e2_entity=['Gene', 'Gene_Family', 'Box', 'Promoter'])
    event_dir['Composes_Primary_Structure'] = event15
    event16 = Event(event_class = 'Compositon',event_name = 'Composes_Protein_Complex',
                  e1_name='Amino_Acid_Sequence',e1_entity=['Protein','Protein_Family',
                                                           'Protein_Complex','Protein_Domain'],
                  e2_name='Protein_Complex',e2_entity=['Protein_Complex'])
    event_dir['Composes_Protein_Complex'] = event16
    event17 = Event(event_class = 'Compositon',event_name = 'Is_Member_Of_Family',
                  e1_name='Element',e1_entity=['Gene', 'Gene_Family', 'RNA', 'Protein', 
                                               'Protein_Family', 'Protein_Domain'],
                  e2_name='Family',e2_entity=['Gene_Family', 'RNA', 'Protein_Family'])
    event_dir['Is_Member_Of_Family'] = event17
    event18 = Event(event_class = 'Compositon',event_name = 'Is_Protein_Domain_Of',
                  e1_name='Domain',e1_entity=['Protein_Domain'],
                  e2_name='Product',e2_entity= ['RNA', 'Protein', 'Protein_Family', 'Protein_Complex', 
                                                'Protein_Domain'])
    event_dir['Is_Protein_Domain_Of'] = event18
    event19 = Event(event_class = 'Compositon',event_name = 'Has_Sequence_Identical_To',
                  e1_name='Domain',e1_entity=all_entities,
                  e2_name='Product',e2_entity=all_entities)
    event_dir['Has_Sequence_Identical_To'] = event19
    ## interactions
    event20 = Event(event_class = 'Interaction',event_name = 'Interacts_With',
                  e1_name='Agent',e1_entity=['Gene', 'Gene_Family', 'Box', 'Promoter','Gene', 'Gene_Family', 
                                             'Box', 'Promoter', 'Protein', 'Protein_Family', 'Protein_Complex', 
                                             'Protein_Domain'],
                  e2_name='Target',e2_entity=['Gene', 'Gene_Family', 'Box', 'Promoter','Gene', 'Gene_Family', 
                                              'Box', 'Promoter', 'Protein', 'Protein_Family', 'Protein_Complex', 
                                              'Protein_Domain'])
    event_dir['Interacts_With'] = event20
    event21 = Event(event_class = 'Interaction',event_name = 'Binds_To',
                  e1_name='Funtional_molecule',e1_entity=['RNA','Protein','Protein_Family',
                                                          'Protein_Complex','Protein_Domain',
                                                          'Hormone'],
                  e2_name='Molecule',e2_entity=['Gene', 'Gene_Family', 'Box', 'Promoter', 'RNA', 
                                                'Protein', 'Protein_Family', 'Protein_Complex', 
                                                'Protein_Domain', 'Hormone'])
    event_dir['Binds_To'] = event21
    ###
    event22 = Event(event_class = 'Linked_to',event_name = 'Is_Linked_To',
                  e1_name='Agent',e1_entity=all_entities,
                  e2_name='Target',e2_entity=all_entities)
    event_dir['Is_Linked_To'] = event22
    return event_dir
    

RELATIONS = ['Is_Member_Of_Family','Is_Linked_To','Regulates_Tissue_Development',
             'Has_Sequence_Identical_To','Is_Protein_Domain_Of','Regulates_Process',
             'Composes_Protein_Complex','Occurs_During','Interacts_With','Is_Localized_In',
             'Composes_Primary_Structure','Occurs_In_Genotype','Binds_To','Exists_In_Genotype',
             'Regulates_Accumulation','Transcribes_Or_Translates_To','Regulates_Development_Phase',
             'Regulates_Expression','Regulates_Molecule_Activity','Is_Functionally_Equivalent_To',
             'Exists_At_Stage','Is_Involved_In_Process']


RELATIONS_21 = ['Is_Member_Of_Family','Is_Linked_To','Regulates_Tissue_Development',
                'Is_Protein_Domain_Of','Regulates_Process',
                'Composes_Protein_Complex','Occurs_During','Interacts_With','Is_Localized_In',
                'Composes_Primary_Structure','Occurs_In_Genotype','Binds_To','Exists_In_Genotype',
                'Regulates_Accumulation','Transcribes_Or_Translates_To','Regulates_Development_Phase',
                'Regulates_Expression','Regulates_Molecule_Activity',
                'Exists_At_Stage','Is_Involved_In_Process']


