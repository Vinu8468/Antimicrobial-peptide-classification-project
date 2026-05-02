from Bio import SeqIO
import pandas as pd

# giving the path to the FASTA
file_path = "../data/raw/plantAMPs_APD2024.fasta"

def load_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path,"fasta"):
        sequences.append({
            "id": record.id,
            "sequence":str(record.seq)
        })
    return pd.DataFrame(sequences)

def main():
    df = load_fasta(file_path)

    # add label (AMP =1 ) so that non amp can be 0
    df["label"] = 1

    # add sequence length
    df["length"]= df["sequence"].apply(len)

    print("Total sequences:",len(df))
    print(df.head())

    # save processed data
    df.to_csv("../data/processed/plant_amp.csv",index=False)
    
if __name__ =="__main__":
    main()
