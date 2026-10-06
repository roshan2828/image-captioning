import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# -------- ENCODER (PRETRAINED RESNET18) --------
class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(pretrained=True)

        # remove avgpool + fc
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # reduce channels 512 → 128 (for decoder)
        self.conv1x1 = nn.Conv2d(512, 128, kernel_size=1)

    def forward(self, images):
        features = self.backbone(images)      # (B,512,7,7)
        features = self.conv1x1(features)     # (B,128,7,7)
        features = features.permute(0,2,3,1)  # (B,7,7,128)
        return features


# -------- ATTENTION --------
class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, features, hidden):
        B,H,W,C = features.size()

        hidden = hidden.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        concat = torch.cat((features, hidden), dim=-1)

        energy = torch.tanh(self.attn(concat))
        scores = self.v(energy).squeeze(-1)

        alpha = F.softmax(scores.view(B,-1), dim=1)

        context = torch.sum(
            features.view(B,-1,C) * alpha.unsqueeze(2),
            dim=1
        )

        return context, alpha


# -------- DECODER --------
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, feature_dim=128):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.attention = Attention(feature_dim, hidden_dim)

    def forward(self, features, captions):
        batch_size = features.size(0)
        max_len = captions.size(1)

        device = features.device
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        cell = torch.zeros(batch_size, self.hidden_dim, device=device)

        outputs = []

        for t in range(max_len):
            context,_ = self.attention(features, hidden)
            embed = self.embedding(captions[:,t])

            lstm_input = torch.cat((embed, context), dim=1)
            hidden,cell = self.lstm(lstm_input,(hidden,cell))

            out = self.fc(hidden)
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)


    def generate(self, features, vocab, max_len=20):
        inv_vocab = {v:k for k,v in vocab.items()}

        batch_size = features.size(0)

        device = features.device
        hidden = torch.zeros(batch_size,self.hidden_dim,device=device)
        cell   = torch.zeros(batch_size,self.hidden_dim,device=device)

        word = torch.tensor([vocab["<start>"]], device=device)

        caption_ids = []

        for _ in range(max_len):
            context,_ = self.attention(features, hidden)
            embed = self.embedding(word)

            lstm_input = torch.cat((embed, context), dim=1)
            hidden,cell = self.lstm(lstm_input,(hidden,cell))

            scores = self.fc(hidden)
            predicted = scores.argmax(dim=1)

            idx = predicted.item()
            caption_ids.append(idx)

            if inv_vocab.get(idx) == "<end>":
                break

            word = predicted

        return caption_ids


# -------- FULL MODEL --------
class ImageCaptionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, vocab, max_len=20):
        features = self.encoder(image)
        return self.decoder.generate(features, vocab, max_len)
