import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Configuración (ajusta J y FPS a tu dataset)
# ------------------------------------------------------------
J = 17                 # nº de articulaciones
D_IN_PER_JOINT = 2     # (x, y)
FPS_DEFAULT = 50       # para calcular vx/ax por segundo (ajústalo)
D_MODEL = 256
H = 8
DEPTH = 4
D_FF = 4 * D_MODEL     # Num neuronas de las capas linear entre bloques de atencion
DROPOUT = 0.1
MAX_LEN = 4096         # Num frames maximo que se cubren 

# ------------------------------------------------------------
# Embedding temporal: Positional Encoding (seno-cos, fijo)
# ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, L, d)
        L = x.size(1)
        return x + self.pe[:, :L]

# ------------------------------------------------------------
# Embedding por articulación: R^{din} -> R^{d_model}
# (compartida para todas las articulaciones)
# ------------------------------------------------------------
class PerJointProjector(nn.Module):
    def __init__(self, in_per_joint: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(d_model, 2 * d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(in_per_joint),
            nn.Linear(in_per_joint, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):  # x: (B, T, J, in_per_joint)
        B, T, J_, C = x.shape
        x = x.view(B * T * J_, C)
        x = self.net(x)
        return x.view(B, T, J_, -1)  # (B,T,J,d)

# ------------------------------------------------------------
# Utilidades de dinámica (velocidad y aceleración)
# ------------------------------------------------------------
@torch.no_grad()
def finite_diff_first(x, dim=1):
    """Primera derivada con padding por delante (replica el primer frame).
    x: tensor (..., T, ...) -> mismo shape.
    """
    pad = x.index_select(dim, torch.tensor([0], device=x.device))
    return torch.diff(x, dim=dim, prepend=pad)

@torch.no_grad()
def add_dynamics(x_xy, fps: int):
    """De (B,T,J,2) -> concat (x,y,vx,vy,ax,ay) = (B,T,J,6)."""
    vxvy = finite_diff_first(x_xy, dim=1) * float(fps)
    axay = finite_diff_first(vxvy, dim=1) * float(fps)
    return torch.cat([x_xy, vxvy, axay], dim=-1)


'''
@torch.no_grad() desactiva el cálculo y almacenamiento de gradientes dentro de la 
función, haciendo el código más eficiente cuando no necesitas entrenamiento.
'''
# ------------------------------------------------------------
# Transformer Encoder 
# ------------------------------------------------------------
def make_encoder(d_model=D_MODEL, n_heads=H, d_ff=D_FF, dropout=DROPOUT, depth=DEPTH):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_ff,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth)

# ------------------------------------------------------------
# Modelo principal
#   Entrada esperada: x en forma (B, T, J, 2) con (x,y) normalizados.
#   lengths: longitudes reales (sin pad) por secuencia (B,)
# ------------------------------------------------------------
class PoseSequenceScorer(nn.Module):
    def __init__(self,
                 num_joints: int = J,
                 in_per_joint: int = D_IN_PER_JOINT,   # 2 (x,y)
                 use_dynamics: bool = True,
                 fps: int = FPS_DEFAULT,
                 d_model: int = D_MODEL,
                 n_heads: int = H,
                 depth: int = DEPTH,
                 d_ff: int = D_FF,
                 dropout: float = DROPOUT,
                 max_len: int = MAX_LEN,
                 use_cls: bool = True,
                 attention_pool: bool = True):
        super().__init__()
        self.J = num_joints
        self.use_dynamics = use_dynamics
        self.fps = fps
        din = in_per_joint * (3 if use_dynamics else 1)  # (x,y) * [x,y | vx,vy | ax,ay]

        # Proyección por articulación
        self.proj = PerJointProjector(din, d_model, dropout)

        # Embedding de articulación (lookup por ID 0..J-1)
        self.joint_embed = nn.Embedding(num_embeddings=num_joints, embedding_dim=d_model)

        # PE temporal (se añade tras aplanar T*J)
        self.pos = PositionalEncoding(d_model, max_len)

        # Encoder
        self.encoder = make_encoder(d_model, n_heads, d_ff, dropout, depth)

        # Token CLS opcional
        self.use_cls = use_cls
        if use_cls:
            self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Attention pooling temporal opcional (mejor que sólo CLS)
        self.attention_pool = attention_pool
        if attention_pool:
            self.pool = nn.Linear(d_model, 1)

        # Heads de salida (ejemplo compatible con tu archivo original)
        D = d_model
        self.head_rotation = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 25))
        self.head_frontback = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2))
        self.head_position3 = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 3))
        self.head_block = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 7)) for _ in range(6)
        ])

    def forward(self, x, lengths=None):
        """
        x: (B, T, J, 2)  -> posiciones 2D normalizadas por joint
        lengths: (B,) longitudes reales sin padding (opcional)
        """
        assert x.dim() == 4 and x.size(2) == self.J and x.size(-1) == 2, \
            "x debe ser (B,T,J,2). Si tienes (B,T,F), reordena a (B,T,J,2)."

        B, T, J, _ = x.shape

        # ---- Dinámica (vx,vy,ax,ay) ----
        if self.use_dynamics:
            x = add_dynamics(x, fps=self.fps)  # (B,T,J,6)

        # ---- Proyección por articulación ----
        h = self.proj(x)  # (B,T,J,d)

        # ---- Embedding de ID de articulación (lookup) ----
        joint_ids = torch.arange(self.J, device=x.device)                 # (J,)
        joint_emb = self.joint_embed(joint_ids)                           # (J,d)
        joint_emb = joint_emb.view(1, 1, J, -1)                           # (1,1,J,d)
        h = h + joint_emb                                                 # (B,T,J,d)

        # ---- Flatten a secuencia (T*J) y PE temporal ----
        seq = h.view(B, T * J, -1)                                        # (B, T*J, d) tokens = joins en frame
        seq = self.pos(seq)                                               # + PE temporal (repetida por J)

        # ---- Máscara de padding ----
        key_padding_mask = None
        if lengths is not None:
            # Máscara temporal (B, T) -> repite cada timestep J veces -> (B, T*J)
            ar = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)  # (B,T)
            valid = ar < lengths.long().unsqueeze(1)                           # (B,T)
            pad_t = ~valid                                                     # True = PAD
            pad = pad_t.repeat_interleave(J, dim=1)                            # (B, T*J) (validamos articulaciones 
                                                                                # en los frames validos)

            if self.use_cls:
                # Añade una columna False (no enmascarar) para CLS al principio
                cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
                key_padding_mask = torch.cat([cls_pad, pad], dim=1)            # (B, 1+T*J)
            else:
                key_padding_mask = pad                                         # (B, T*J)

        # ---- Inserta CLS si procede ----
        if self.use_cls:
            cls_tok = self.cls.expand(B, 1, -1)                                # (B,1,d)
            seq = torch.cat([cls_tok, seq], dim=1)                             # (B, 1+T*J, d)

        # ---- Encoder ----
        enc = self.encoder(seq, src_key_padding_mask=key_padding_mask)         # (B, L, d)

        # ---- Representación global ----
        if self.attention_pool:
            # Ignora CLS para el pooling y aplica máscara
            if self.use_cls:
                enc_no_cls = enc[:, 1:, :]                                     # (B, T*J, d)
                pad_mask_no_cls = key_padding_mask[:, 1:] if key_padding_mask is not None else None
            else:
                enc_no_cls = enc
                pad_mask_no_cls = key_padding_mask

            # Scores de atención por timestep*joint
            w = self.pool(enc_no_cls).squeeze(-1)                              # (B, T*J)
            if pad_mask_no_cls is not None:
                w = w.masked_fill(pad_mask_no_cls, float('-inf'))
            attn = torch.softmax(w, dim=1)                                     # (B, T*J)
            rep = torch.einsum('bld,bl->bd', enc_no_cls, attn)                 # (B, d)

            # (Opcional) concatena CLS si quieres
            if self.use_cls:
                rep = torch.cat([enc[:, 0], rep], dim=-1)                      # (B, 2d)
        else:
            rep = enc[:, 0] if self.use_cls else enc.mean(dim=1)               # (B, d)

        # ---- Heads de salida de ejemplo (ajusta a tu tarea) ----
        D_out = rep.size(-1)
        # Si concatenaste CLS+pool tendrás 2d; adapta las heads:
        def head_maker(out_layer, d_in):
            # Actualiza las Linear si necesitas 2d
            if isinstance(out_layer, nn.Sequential):
                ln, fc = out_layer
                if fc.in_features != d_in:
                    out_layer[1] = nn.Linear(d_in, fc.out_features)
            return out_layer

        self.head_rotation = head_maker(self.head_rotation, D_out)
        self.head_frontback = head_maker(self.head_frontback, D_out)
        self.head_position3 = head_maker(self.head_position3, D_out)
        for i in range(len(self.head_block)):
            self.head_block[i] = head_maker(self.head_block[i], D_out)

        out = {
            "rotation_quarter_logits": self.head_rotation(rep),   # (B,25)
            "front_back_logits": self.head_frontback(rep),        # (B,2)
            "position3_logits": self.head_position3(rep),         # (B,3)
            "x_numbers_logits": torch.stack(
                [head(rep) for head in self.head_block], dim=1     # (B,6,7)
            )
        }
        return out

# -----------------------------------------------------------------
# Nota sobre device=x.device:
# En PyTorch, todo tensor vive en un dispositivo (CPU o GPU). 
# Cuando creas tensores nuevos (arange, zeros, etc.), usa device=x.device
# para ponerlos en el MISMO dispositivo que x y evitar errores del tipo:
#   RuntimeError: Expected all tensors to be on the same device...
# -----------------------------------------------------------------
