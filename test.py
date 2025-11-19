# import pandas as pd
# print("hello world")
# df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# ===========================
# English → Vietnamese Seq2Seq with Attention
# Clean Full Version (Updated)
# ===========================

from __future__ import unicode_literals, print_function, division
import unicodedata
import regex as re
import random
import time
import math

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# Special Tokens & Config
# ===========================
PAD_token = 0
SOS_token = 1
EOS_token = 2

MAX_LENGTH = 60   # max tokens per sentence (after truncation)


# ===========================
# Vocabulary Class
# ===========================
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
        }
        self.n_words = 3  # PAD, SOS, EOS

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            if word.strip():
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# ===========================
# Normalization
# ===========================
def normalizeString(s):
    s = s.lower().strip()
    s = s.replace("…", ".")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"([.!?])", r" \1 ", s)
    s = re.sub(r"[^\p{L}\p{N}.!?']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ===========================
# Load Dataset
# ===========================
def readLangs(path):
    print("Reading dataset:", path)
    # File already has header row; keep it and rename columns
    df = pd.read_csv(path, sep="\t")
    df.columns = ["id_en", "en", "id_vi", "vi"]
    df = df.dropna()

    pairs = []
    for _, row in df.iterrows():
        en = normalizeString(row["en"])
        vi = normalizeString(row["vi"])
        pairs.append([en, vi])

    return pairs


def prepareData(path):
    pairs = readLangs(path)

    input_lang = Lang("eng")
    output_lang = Lang("vie")

    for en, vi in pairs:
        input_lang.addSentence(en)
        output_lang.addSentence(vi)

    print("Total sentence pairs:", len(pairs))
    print("Input vocab size:", input_lang.n_words)
    print("Output vocab size:", output_lang.n_words)

    return input_lang, output_lang, pairs


# ===========================
# Sentence → Tensor
# ===========================
def indexesFromSentence(lang, sentence):
    # Unknown words mapped to PAD_token (you can add UNK if you want)
    return [lang.word2index.get(word, PAD_token) for word in sentence.split(" ")]

def tensorFromSentence(lang, sentence):
    idxs = indexesFromSentence(lang, sentence)
    idxs = idxs[:MAX_LENGTH - 1]
    idxs.append(EOS_token)
    if len(idxs) < MAX_LENGTH:
        idxs += [PAD_token] * (MAX_LENGTH - len(idxs))
    return torch.tensor(idxs, dtype=torch.long, device=device).unsqueeze(0)  # (1, MAX_LENGTH)


# ===========================
# Build Dataloader
# ===========================
def get_dataloader(batch_size, path):
    input_lang, output_lang, pairs = prepareData(path)

    n = len(pairs)
    input_ids = np.full((n, MAX_LENGTH), PAD_token, dtype=np.int32)
    target_ids = np.full((n, MAX_LENGTH), PAD_token, dtype=np.int32)

    for idx, (en, vi) in enumerate(pairs):
        en_ids = indexesFromSentence(input_lang, en)[:MAX_LENGTH - 1]
        vi_ids = indexesFromSentence(output_lang, vi)[:MAX_LENGTH - 1]

        en_ids.append(EOS_token)
        vi_ids.append(EOS_token)

        input_ids[idx, :len(en_ids)] = en_ids
        target_ids[idx, :len(vi_ids)] = vi_ids

    dataset = TensorDataset(
        torch.LongTensor(input_ids).to(device),
        torch.LongTensor(target_ids).to(device),
    )

    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    return input_lang, output_lang, pairs, dataloader


# ===========================
# Encoder GRU
# ===========================
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_tensor):
        # input_tensor: (batch, seq_len)
        embedded = self.embedding(input_tensor)     # (batch, seq_len, hidden)
        outputs, hidden = self.gru(embedded)        # outputs: (batch, seq_len, hidden)
        return outputs, hidden                      # hidden: (1, batch, hidden)


# ===========================
# Bahdanau Attention
# ===========================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        """
        query: (batch, 1, hidden)
        keys:  (batch, seq_len, hidden)
        """
        # Broadcast add: (batch, seq_len, hidden)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)                            # (batch, seq_len, 1)
        context = torch.sum(weights * keys, dim=1, keepdim=True)      # (batch, 1, hidden)
        return context, weights


# ===========================
# Decoder with Attention
# ===========================
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_token)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        """
        encoder_outputs: (batch, seq_len, hidden)
        encoder_hidden:  (1, batch, hidden)
        target_tensor:   (batch, max_len) or None
        """
        batch_size = encoder_outputs.size(0)
        decoder_hidden = encoder_hidden
        decoder_input = torch.full(
            (batch_size, 1), SOS_token, dtype=torch.long, device=device
        )  # (batch, 1)

        outputs = []

        target_len = target_tensor.size(1) if target_tensor is not None else MAX_LENGTH

        for t in range(target_len):
            # embedded: (batch, 1, hidden)
            embedded = self.embedding(decoder_input)

            # query for attention: last layer hidden state (batch, 1, hidden)
            query = decoder_hidden[-1].unsqueeze(1)
            context, _ = self.attention(query, encoder_outputs)  # (batch, 1, hidden)

            rnn_input = torch.cat((embedded, context), dim=2)    # (batch, 1, 2*hidden)
            rnn_output, decoder_hidden = self.gru(rnn_input, decoder_hidden)
            # rnn_output: (batch, 1, hidden)

            step_output = F.log_softmax(self.out(rnn_output.squeeze(1)), dim=-1)  # (batch, vocab)
            outputs.append(step_output.unsqueeze(1))  # (batch, 1, vocab)

            if target_tensor is not None:
                # Teacher forcing: next input is ground truth
                decoder_input = target_tensor[:, t].unsqueeze(1)  # (batch, 1)
            else:
                # Greedy decoding
                top1 = step_output.argmax(1)          # (batch,)
                decoder_input = top1.unsqueeze(1)     # (batch, 1)

        outputs = torch.cat(outputs, dim=1)           # (batch, target_len, vocab)
        return outputs


# ===========================
# Training Loop
# ===========================
def train_one_epoch(dataloader, encoder, decoder, encoder_opt, decoder_opt, criterion):
    total_loss = 0.0
    encoder.train()
    decoder.train()

    for input_batch, target_batch in dataloader:
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()

        enc_outputs, enc_hidden = encoder(input_batch)
        dec_outputs = decoder(enc_outputs, enc_hidden, target_batch)

        loss = criterion(
            dec_outputs.reshape(-1, dec_outputs.size(-1)),
            target_batch.reshape(-1),
        )
        loss.backward()

        encoder_opt.step()
        decoder_opt.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(train_dataloader, encoder, decoder, epochs):
    encoder_opt = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_opt = optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.NLLLoss(ignore_index=PAD_token)

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(
            train_dataloader, encoder, decoder, encoder_opt, decoder_opt, criterion
        )
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")


# ===========================
# Evaluation
# ===========================
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        norm_sentence = normalizeString(sentence)
        input_tensor = tensorFromSentence(input_lang, norm_sentence)  # (1, MAX_LENGTH)

        enc_outputs, enc_hidden = encoder(input_tensor)
        decoder_hidden = enc_hidden
        decoder_input = torch.tensor([[SOS_token]], device=device)  # (1, 1)

        decoded_words = []

        for _ in range(MAX_LENGTH):
            embedded = decoder.embedding(decoder_input)             # (1, 1, hidden)
            query = decoder_hidden[-1].unsqueeze(1)                 # (1, 1, hidden)
            context, _ = decoder.attention(query, enc_outputs)      # (1, 1, hidden)
            rnn_input = torch.cat((embedded, context), dim=2)       # (1, 1, 2*hidden)

            rnn_output, decoder_hidden = decoder.gru(rnn_input, decoder_hidden)
            output = F.log_softmax(decoder.out(rnn_output.squeeze(1)), dim=-1)  # (1, vocab)

            top1 = output.argmax(1).item()

            if top1 == EOS_token:
                decoded_words.append("<EOS>")
                break
            if top1 == PAD_token:
                break

            decoded_words.append(output_lang.index2word.get(top1, "<UNK>"))
            decoder_input = torch.tensor([[top1]], device=device)

    return " ".join(decoded_words)


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    data_path = "Sentence pairs in English-Vietnamese - 2025-11-12.tsv"

    batch_size = 16
    hidden_size = 128

    input_lang, output_lang, pairs, dataloader = get_dataloader(batch_size, data_path)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    train_model(dataloader, encoder, decoder, epochs=5)

    # quick test
    test_sentence = "how are you today ?"
    print("EN:", test_sentence)
    print("VI:", evaluate(encoder, decoder, test_sentence, input_lang, output_lang))
