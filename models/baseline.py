from torch import nn
from torch_geometric.nn import DimeNetPlusPlus

class DumplingA1(nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        out_channels=1,
        cutoff=5.0,
        max_num_neighbors=32,
        num_blocks=3,
    ):
        super().__init__()
        # DimeNet++ специфичен: он сам считает углы и расстояния внутри
        self.gnn = DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            out_channels=hidden_channels, # Выход в латентное пространство
            num_blocks=num_blocks,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=128,
            num_spherical=7,
            num_radial=6,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=5
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, batch_list, progress=0.0):
        # z: атомные номера (batch.x[:, 0])
        # pos: координаты
        # batch: индекс батча
        prot_data, lig_data, pock_data, complex_data, target = batch_list
        z = complex_data.x[:, 0].long()
        pos = complex_data.pos.float()
        
        # Получаем эмбеддинг всей системы
        x = self.gnn(z, pos, complex_data.batch)
        
        # Предсказываем Delta G
        return self.head(x).view(-1)
