#!/bin/env python3

import sys
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="""Split examples in train/test/dev parts with proportion.""")
    parser.add_argument("-i", "--infile", type=str, required=True, help="""TAB separated 2-column file.""")
    args = parser.parse_args()
    
    dataset = pd.read_csv(args.infile, sep='\t', header=None, names=['Class', 'Txt'], dtype=str)
    
    minval = dataset.groupby('Class').size().min()
    numintrain = (minval * 8) // 10
    numintest = minval // 10
    
    print(f"Examples per class: {minval}")
    print(f"Expected examples in train per class: {numintrain}")
    print(f"Expected examples in val and test per class: {numintest}")
    
    def split_group(group):
        train = group.iloc[:numintrain]
        test = group.iloc[numintrain:numintrain + numintest]
        dev = group.iloc[numintrain + numintest:numintrain + 2 * numintest]
        return train, test, dev
        
    train_list, test_list, dev_list = [], [], []
    
    for _, group in dataset.groupby('Class'):
        train, test, dev = split_group(group)
        train_list.append(train)
        test_list.append(test)
        dev_list.append(dev)
        
    traindataset = pd.concat(train_list)
    testdataset = pd.concat(test_list)
    devdataset = pd.concat(dev_list)
    
    print(f"Total examples in train: {traindataset.size}")
    print(f"Total examples in val: {devdataset.size/2}")
    print(f"Total examples in test: {testdataset.size/2}")
    
    namenoext = args.infile.replace(".tsv","")
    traindataset.to_csv(f"{namenoext}_train.tsv", sep='\t', mode='w', encoding='utf-8', index=False, header=False)
    testdataset.to_csv(f"{namenoext}_test.tsv", sep='\t', mode='w', encoding='utf-8', index=False, header=False)
    devdataset.to_csv(f"{namenoext}_dev.tsv", sep='\t', mode='w', encoding='utf-8', index=False, header=False)
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
