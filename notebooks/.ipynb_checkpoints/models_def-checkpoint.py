{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "114e68eb-258c-4515-b6ee-b4b7f6c3d868",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4ad100b9-d8bf-4cdd-a285-f457a6dbf616",
   "metadata": {},
   "outputs": [],
   "source": [
    "class SumCalibBaseline(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.a = nn.Parameter(torch.tensor(1.0))\n",
    "        self.b = nn.Parameter(torch.tensor(0.0))\n",
    "\n",
    "    def forward(self, tokens, mask):\n",
    "        e = tokens[..., 3] * mask.to(tokens.dtype)\n",
    "        s = e.sum(dim=1)\n",
    "        return self.a * s + self.b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "21e966a5-5210-450e-9065-a57de4f9f53d",
   "metadata": {},
   "outputs": [],
   "source": [
    "class TokenEmbed(nn.Module):\n",
    "    def __init__(self, d_model: int, dropout: float = 0.1):\n",
    "        super().__init__()\n",
    "        self.coord_mlp = nn.Sequential(\n",
    "            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)\n",
    "        )\n",
    "        self.energy_mlp = nn.Sequential(\n",
    "            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)\n",
    "        )\n",
    "        self.norm = nn.LayerNorm(d_model)\n",
    "        self.drop = nn.Dropout(dropout)\n",
    "\n",
    "    def forward(self, tokens):\n",
    "        xy = tokens[..., 0:2]\n",
    "        le = tokens[..., 2:4]\n",
    "        x = self.coord_mlp(xy) + self.energy_mlp(le)\n",
    "        return self.drop(self.norm(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cca642bb-7f03-4dc6-88b7-5b1800e3a245",
   "metadata": {},
   "outputs": [],
   "source": [
    "class TransformerRegressor(nn.Module):\n",
    "    def __init__(self, d_model=96, n_heads=4, n_layers=3, dropout=0.1, mlp_ratio=4, pooling=\"cls\"):\n",
    "        super().__init__()\n",
    "        self.pooling = pooling\n",
    "        self.embed = TokenEmbed(d_model=d_model, dropout=dropout)\n",
    "\n",
    "        if pooling == \"cls\":\n",
    "            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))\n",
    "            nn.init.normal_(self.cls, std=0.02)\n",
    "        else:\n",
    "            self.cls = None\n",
    "\n",
    "        enc_layer = nn.TransformerEncoderLayer(\n",
    "            d_model=d_model, nhead=n_heads,\n",
    "            dim_feedforward=int(d_model * mlp_ratio),\n",
    "            dropout=dropout, activation=\"gelu\",\n",
    "            batch_first=True, norm_first=True\n",
    "        )\n",
    "        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)\n",
    "\n",
    "        self.head = nn.Sequential(\n",
    "            nn.LayerNorm(d_model),\n",
    "            nn.Linear(d_model, d_model),\n",
    "            nn.GELU(),\n",
    "            nn.Dropout(dropout),\n",
    "            nn.Linear(d_model, 1),\n",
    "        )\n",
    "\n",
    "    def forward(self, tokens, mask):\n",
    "        x = self.embed(tokens)\n",
    "        pad_mask = ~mask\n",
    "\n",
    "        if self.pooling == \"cls\":\n",
    "            B = x.size(0)\n",
    "            cls = self.cls.expand(B, -1, -1)\n",
    "            x = torch.cat([cls, x], dim=1)\n",
    "            cls_pad = torch.zeros((B, 1), device=pad_mask.device, dtype=pad_mask.dtype)\n",
    "            pad_mask = torch.cat([cls_pad, pad_mask], dim=1)\n",
    "\n",
    "        x = self.encoder(x, src_key_padding_mask=pad_mask)\n",
    "\n",
    "        if self.pooling == \"cls\":\n",
    "            pooled = x[:, 0]\n",
    "        else:\n",
    "            m = mask.to(x.dtype).unsqueeze(-1)\n",
    "            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)\n",
    "\n",
    "        return self.head(pooled).squeeze(-1)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
