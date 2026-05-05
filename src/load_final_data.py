from Bio import SeqIO
import pandas as pd


def load_fasta(file_path,label,min_len=10,max_len=50):
    sequences = []
    for record in SeqIO.parse(file_path,"fasta"):
        seq = str(record.seq)
        length = len(seq)

        # filtering by length
        if min_len <= length <= max_len:
            sequences.append({
                "id": record.id,
                "sequence": seq,
                "label": label,
                "length": length
            })
    return pd.DataFrame(sequences)


def main():
    plant_file = "../data/raw/naturalAMPs_APD2024a.fasta"
    uniprot_file = "../data/raw/peptide_10_50_uniprot_no_toxin_antimicrobial_secreted.fasta"

    print("Loading AMP dataset...")
    amp_df = load_fasta(plant_file,label=1)
    print("Loading non-AMP dataset...")
    non_amp_df = load_fasta(uniprot_file,label=0)

    # Optional: remove toxin-like sequences (basic filter)
    non_amp_df = non_amp_df[
        ~non_amp_df["id"].str.lower().str.contains("toxin", na=False)
    ]
    print("Before balancing:")
    print("AMP count:", len(amp_df))
    print("Non-AMP count:", len(non_amp_df))

    min_size = min(len(amp_df), len(non_amp_df))

    amp_df = amp_df.sample(min_size, random_state=42)
    non_amp_df = non_amp_df.sample(min_size, random_state=42)
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

    def is_valid(seq):
        return all(aa in VALID_AA for aa in seq)

    amp_df = amp_df[amp_df["sequence"].apply(is_valid)]
    non_amp_df = non_amp_df[non_amp_df["sequence"].apply(is_valid)]
    # Combine
    df = pd.concat([amp_df, non_amp_df], ignore_index=True)

    print("\nAfter balancing:")
    print(df["label"].value_counts())
    
    # save the dataset
    output_path = "../data/processed/final_dataset_v2.csv"
    df.to_csv(output_path, index=False)

    print("Dataset saved to:", output_path)

    # basic statistics
    print("Length statistics:")
    print(df.groupby("label")["length"].describe())

if __name__ == "__main__":
    main()
