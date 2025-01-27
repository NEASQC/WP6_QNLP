#!/bin/env python3

import sys
import argparse
import csv
import re
import pandas as pd

def main():

    parser = argparse.ArgumentParser(description="""Balance classes by same number of examples.""")
    parser.add_argument("-i", "--infile", type=str, required=True, help="""TAB separated 2-column file to filter.""")
    parser.add_argument("-o", "--outfile", type=str, required=True, help="""Filtered 2-column file.""")
    parser.add_argument("-c", "--classes", type=str, required=False, help="""Wich classes to ignore""")
    args = parser.parse_args()

    
    dataset = pd.read_csv(args.infile, sep='\t',header=None, names=['Class','Txt'], dtype=str)
    #Randomize data
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    if args.classes:
        classes=args.classes.split(',')
        print(classes)
        dataset=dataset[~dataset['Class'].isin(classes)].reset_index()
        
    minval=dataset.groupby('Class').size().min()
    print(f"Examples in class: {minval}")
    
    def truncate(group):
        if len(group) > minval:
            return group.head(minval)
        return group
    newdataset = dataset.groupby('Class').apply(truncate).reset_index(drop=True)
    newdataset.to_csv(args.outfile, sep='\t', mode='w',encoding='utf-8',index=False,header=False,columns=['Class','Txt'])
    

if __name__ == "__main__":
    sys.exit(int(main() or 0))
