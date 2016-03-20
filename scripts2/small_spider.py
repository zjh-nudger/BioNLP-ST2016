# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:18:38 2016

@author: tanfan.zjh
"""
from Bio_Eutils import Entrez, Medline
Entrez.email = "me@myself.org"

# Search for PMIDs from author text based search
handle = Entrez.esearch(db="pubmed", retmax=100000, term="Arabidopsis Thaliana")
pub_search = Entrez.read(handle)
handle.close()
print pub_search
'''
# Fetch matching entries
handle = Entrez.efetch(db='pubmed', id=pub_search['IdList'], retmax=20, rettype="medline", retmode="text")
pub_items = Medline.parse(handle)

# Work with it
for pub_item in pub_items:
    print "*" * 10
    print "%s - %s." % (
        pub_item.get("TI","?"),
        ", ".join(pub_item.get("AB","?"))
        )

handle.close()
'''