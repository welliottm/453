#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 20:37:59 2018

@author: paulhuynh
"""

import os
import docx2txt
import pandas as pd


#set working directory
os.chdir('/Users/paulhuynh/Documents/School/MSDS 453/Class Corpus')

#function to retreive and turn document into text
def retrieve_DSI(file_name):
    file_name=str(file_name)
    text = docx2txt.process(file_name)
    return text

#Lists to store file name and body of text
file_name=[]
text=[]

#for loop to iterate through documents in working directory
for file in os.listdir('.'):
    #if statment to not attempt to open non word documents
    if file.endswith('.docx'):
        text_name=file
        #call function to obtain the text
        text_body=retrieve_DSI(file)
        #apped the file names and text to list
        file_name.append(text_name)
        text.append(text_body)
        #removed the variables used in the for loop
        del text_name, text_body, file

#create dictionary for corpus
corpus={'DSI_Title':file_name, 'Text': text}


#output a CSV with containing the class corpus along with titles of corpus.  
#file saved in working directory.
pd.DataFrame(corpus).to_csv('Class Corpus.csv', index=file_name)
