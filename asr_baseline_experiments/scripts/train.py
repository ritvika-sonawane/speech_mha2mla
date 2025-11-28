"""
Training script for ASR models with different attention mechanisms
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path

# Add models to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.conformer import build_conformer
from models.branchformer import build_branchformer


class LibriSpeechDataset(torch.utils.data.Dataset):
    """LibriSpeech dataset wrapper"""
    
    def __init__(self, split='train', data_dir='data/librispeech', quick_test=False):
        self.data_dir = data_dir
        
        # Load using HuggingFace datasets
        if split == 'train':
            if quick_test:
                # Use only 1% of data for quick testing on CPU
                self.dataset = load_dataset("librispeech_asr", "clean", split="train.100[:1%]")
            else:
                self.dataset = load_dataset("librispeech_asr", "clean", split="train.100")
        elif split == 'test-clean':
            if quick_test:
                self.dataset = load_dataset("librispeech_asr", "clean", split="test[:1%]")
            else:
                self.dataset = load_dataset("librispeech_asr", "clean", split="test")
        elif split == 'test-other':
            self.dataset = load_dataset("librispeech_asr", "other", split="test")
        
        # Build vocabulary from training set
        self.vocab = self._build_vocab()
        
    def _build_vocab(self):
        """Build character-level vocabulary"""
        chars = set()
        for item in self.dataset:
            chars.update(item['text'].lower())
        
        vocab = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        
        for i, char in enumerate(sorted(chars), start=4):
            vocab[char] = i
        
        return vocab
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Audio
        audio = torch.tensor(item['audio']['array']).float()
        
        # Text to indices
        text = item['text'].lower()
        text_indices = [self.vocab.get(c, self.vocab['<unk>']) for c in text]
        text_indices = [self.vocab['<sos>']] + text_indices + [self.vocab['<eos>']]
        
        return audio, torch.tensor(text_indices), len(audio), len(text_indices)


def collate_fn(batch):
    """Collate function for batching"""
    audios, texts, audio_lens, text_lens = zip(*batch)
    
    # Pad audios
    max_audio_len = max(audio_lens)
    padded_audios = torch.zeros(len(audios), max_audio_len)
    for i, audio in enumerate(audios):
        padded_audios[i, :len(audio)] = audio
    
    # Pad texts
    max_text_len = max(text_lens)
    padded_texts = torch.zeros(len(texts), max_text_len, dtype=torch.long)
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = text
    
    return (
        padded_audios,
        padded_texts,
        torch.tensor(audio_lens),
        torch.tensor(text_lens)
    )


def extract_features(audio, n_mels=80):
    """Extract mel-spectrogram features"""
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=n_mels,
        n_fft=400,
        hop_length=160
    ).to(audio.device)  # Move transform to same device as audio
    
    features = mel_spec(audio)
    features = torch.log(features + 1e-9)
    return features.transpose(1, 2)  # [B, T, n_mels]


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (audios, texts, audio_lens, text_lens) in enumerate(pbar):
        audios = audios.to(device)
        texts = texts.to(device)
        
        # Extract features
        features = extract_features(audios)
        
        # Calculate feature lengths after mel-spectrogram extraction
        # hop_length=160 means audio_len / 160 frames
        feature_lens = (audio_lens // 160).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(features, feature_lens)
        
        # Compute loss (CTC Loss)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # [T, B, vocab_size]
        
        input_lengths = feature_lens // 4  # Account for model subsampling
        target_lengths = text_lens.to(device)
        
        loss = criterion(log_probs, texts, input_lengths, target_lengths)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for audios, texts, audio_lens, text_lens in tqdm(dataloader, desc="Evaluating"):
            audios = audios.to(device)
            texts = texts.to(device)
            
            # Extract features
            features = extract_features(audios)
            
            # Calculate feature lengths after mel-spectrogram extraction
            feature_lens = (audio_lens // 160).to(device)
            
            # Forward pass
            logits, _ = model(features, feature_lens)
            
            # Compute loss
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)
            
            input_lengths = feature_lens // 4
            target_lengths = text_lens.to(device)
            
            loss = criterion(log_probs, texts, input_lengths, target_lengths)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create datasets
    print("Loading datasets...")
    quick_test = args.quick_test if hasattr(args, 'quick_test') else False
    if quick_test:
        print("⚠️  QUICK TEST MODE: Using only 1% of data")
    train_dataset = LibriSpeechDataset('train', quick_test=quick_test)
    test_dataset = LibriSpeechDataset('test-clean', quick_test=quick_test)
    
    vocab_size = len(train_dataset.vocab)
    config['vocab_size'] = vocab_size
    
    # Save vocabulary
    with open(output_dir / 'vocab.json', 'w') as f:
        json.dump(train_dataset.vocab, f)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Build model
    print(f"Building {config['model_type']} model with {config['attention_type']} attention...")
    if config['model_type'] == 'conformer':
        model = build_conformer(config)
    elif config['model_type'] == 'branchformer':
        model = build_branchformer(config)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    model = model.to(device)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-6)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # TensorBoard
    writer = SummaryWriter(output_dir / 'logs')
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config.get('num_epochs', 50) + 1):
        print(f"\nEpoch {epoch}/{config.get('num_epochs', 50)}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Evaluate
        val_loss = evaluate(model, test_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"Saved best model with val loss: {val_loss:.4f}")
    
    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ASR model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--quick_test', action='store_true', help='Use 1%% of data for quick CPU testing')
    
    args = parser.parse_args()
    main(args)
