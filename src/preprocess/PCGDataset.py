from pathlib import Path

from torch.utils.data import Dataset

import pandas as pd


class PCGDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Args:
            root_dir (str or Path): Path to the parent folder containing the 6 subfolders.
            transform (callable, optional): Transform to apply to the waveform tensor.
            target_transform (callable, optional): Transform to apply to the label.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        self.records = []

        # 1. Find all REFERENCE.csv files in all subfolders
        # This is extremely fast because it only looks for 6 files.
        for csv_path in self.root_dir.rglob("REFERENCE.csv"):
            
            # Get the folder where this CSV is located (e.g., folder_A)
            folder_path = csv_path.parent
            
            # header=None because no header row
            df = pd.read_csv(csv_path, header=None)
            
            # 2. Iterate through the CSV rows to build the dataset index
            for _, row in df.iterrows():
                file_id = str(row[0])  # e.g., 'f0001'
                raw_label = int(row[1])  # e.g., -1 or 1
                
                # Construct the absolute path to the .wav file
                wav_path = folder_path / f"{file_id}.wav"
                
                # Convert label: -1 (Normal) -> 0,  1 (Abnormal) -> 1
                # This is highly recommended for PyTorch loss functions!
                label = 0 if raw_label == -1 else 1
                
                if wav_path.exists():
                    self.records.append((str(wav_path), label))
                else:
                    print(f"Warning: {wav_path} not found. Skipping.")

        print(f"Successfully loaded {len(self.records)} audio files.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        wav_path, label = self.records[idx]

        # torchaudio.load returns: (waveform_tensor, sample_rate)
        # We generally only need the waveform, but some transforms require the sample_rate
        waveform, sample_rate = torchaudio.load(wav_path)

        # Apply transformations if provided
        if self.transform:
            waveform = self.transform(waveform)

        if self.target_transform:
            label = self.target_transform(label)

        return waveform, label