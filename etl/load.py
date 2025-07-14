# load.py - Auto-generated
def write_csv(df, output_path):
    df.to_csv(output_path, index=False)
    print(f" Processed data saved at {output_path}")