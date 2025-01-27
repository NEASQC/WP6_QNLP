import sys
import os
import json
import csv
import subprocess
import spacy
from spacy.tokenizer import Tokenizer
import re
import argparse
import html
import re

def main():
    nlp = spacy.load('en_core_web_sm')
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help = "Comma separated CSV data file")
    parser.add_argument("-o", "--outfile", help = "Result file")
    parser.add_argument("-d", "--delimiter", default=',', help = "Field delimiter symbol")
    parser.add_argument("-c", "--classfield", required=False, help = "Name of class field")
    parser.add_argument("-t", "--txtfield", required=False, help = "Name of text field")
    parser.add_argument("-f", "--firstsentence", action='store_true', required=False, help = "Take the first sentence if the text is longer that 20 tokens")   
    args = parser.parse_args()
    
    if args.classfield != None: 
        classfield = args.classfield
        txtfield = args.txtfield
    else:
        classfield = 'f1'
        txtfield = 'f2'
    
    print(args)
    print(classfield+' and '+txtfield)
    with open(args.infile, encoding="utf8", newline='') as csvfile, open(args.outfile, "w", encoding="utf8") as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        if args.classfield != None: 
            news_reader = csv.DictReader(csvfile, delimiter=args.delimiter, quotechar='"')
        else:
            news_reader = csv.DictReader(csvfile, delimiter=args.delimiter, fieldnames = [classfield, txtfield], quotechar='"')        
        processed_summaries = set()
        norm_process_params = ["perl", "./data/data_processing/scripts/normalize-punctuation.perl","-b","-l", "en"]
        norm_process = subprocess.Popen(norm_process_params, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
        
        
        for row in news_reader:
            score = row[classfield]
            if score == '0':
                continue
            if row[txtfield] == None:
                continue
            summary = row[txtfield].replace("\n"," ").replace("\t"," ").replace("\r"," ").replace("\\$","$")
            summary = summary.replace(" quot;","\"").replace(" amp;","&").replace(" #39;","'").replace(" #36;","$").replace(" #151;","—").replace(" #146;","’").replace(" #8220;","“").replace(" #8221;","”").replace(" #38;","&").replace(" #8212;","—").replace(" #232;","è").replace(" #149;","•").replace(" #145;","‘").replace(" #233;","é").replace(" #147;","“").replace(" #133;","…").replace(" #148;","”").replace(" #0151;","—").replace(" #64257;","ﬁ").replace(" #8217;","’").replace(" #163;","£").replace(" #038;","&").replace(" #160;"," ").replace(" #8211;","–").replace(" #37;","%").replace(" #153;","™").replace(" #195;","Ã").replace(" #169;","©").replace(" #225;","á").replace(" #91;","[").replace(" #93;","]").replace(" #x2014;","—").replace(" #8482;","™").replace(" #234;","ê").replace(" #8364;","€")
            summary = re.sub('@[^\s]+ ','',summary)
            summary = html.unescape(summary)
            summary = re.sub('<[^<]+>', "", summary)
            summary = re.sub('\([^\)\(]*(AFP|AP|Reuters|press release|Canadian Press|CP|Sports Network|Investor\'s Business Daily|CBS|Update[1-9]|CMC|washingtonpost.com|USATODAY|Los Angeles Times|PTI|SiliconValley\.com|Antara|TechWeb|IsraelNN\.com|newratings\.com|(SPACE|space)\.com|[Ff]orbes\.com)[^\)\(]*\)', "", summary)
            summary = re.sub('( By Reuters|[\-:\|,] (Telegraph By Reuters|AFP|Reuters|(AFP|Reuters) ([Vv]ideo|[Tt]ally|[Pp]oll|[Ss]ays|[Cc]hief)|Reuters\.com|(English\.)?news\.cn|(The )?Jakarta Post|Volume app news|BC News|ResearchAndMarkets\.com|Chinadaily\.com\.cn|Middle East Monitor|Burundi|Commenter Edition|Sentinelassam|channelnews|TheHill|Insider|Telegraph|(The )?Manila Times|ITV News|Gulf News|Daily Sabah|Australian Associated Press)|By Reuters \| Armenian American Reporter)$', "", summary)
            norm_process.stdin.write(summary.encode('utf-8'))
            norm_process.stdin.write('\n'.encode('utf-8'))
            norm_process.stdin.flush()
            norm_summary = norm_process.stdout.readline().decode("utf-8").rstrip()
            doc=nlp(norm_summary)
            tok_summary = " ".join([token.text for token in doc])
            if tok_summary in processed_summaries:
                continue
            sent_type = ''
            processed_summaries.add(tok_summary)
            writer.writerow([score, tok_summary])
        norm_process.kill()

if __name__ == "__main__":
    sys.exit(int(main() or 0))