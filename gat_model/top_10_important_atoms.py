import torch
import numpy as np
import os
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from rdkit import Chem

# Импорты твоих модулей
from dataset import Ligand_Protein_Dataset
from model import Binding_Affinity_Predictor
from config import Config

def extract_top_10_global_importance():
    print("--- [1/5] Подготовка окружения и данных ---")
    config = Config()
    device = torch.device("cpu")
    
    # Загружаем датасет
    dataset = Ligand_Protein_Dataset(config.root, config.data_dir, config.affinity_file)
    
    # Воспроизводим тестовый сплит, чтобы взять тот же комплекс 4x6o
    indices = np.arange(len(dataset))
    np.random.seed(45)
    np.random.shuffle(indices)
    test_idx = indices[int(len(indices)*0.9):]
    
    # Выбираем первый комплекс из теста (в нашем случае это был 4x6o)
    sample_idx = test_idx[0]
    data = dataset[sample_idx]
    pdb_id = data.pdb_id[0] if isinstance(data.pdb_id, list) else data.pdb_id
    
    # Пути к исходным файлам для идентификации атомов через RDKit
    complex_path = os.path.join(config.data_dir, pdb_id)
    lig_path = os.path.join(complex_path, f"{pdb_id}_ligand.mol2")
    prot_path = os.path.join(complex_path, f"{pdb_id}_pocket.pdb")
    
    ligand_mol = Chem.MolFromMol2File(lig_path, sanitize=True, removeHs=False)
    pocket_mol = Chem.MolFromPDBFile(prot_path, sanitize=True, removeHs=False)
    
    if ligand_mol is None or pocket_mol is None:
        print(f"Ошибка: Не удалось загрузить структуры для {pdb_id}")
        return

    num_lig_atoms = ligand_mol.GetNumAtoms()

    print("--- [2/5] Загрузка обученной модели ---")
    model = Binding_Affinity_Predictor(
        in_channels=config.in_channels,
        num_gnn_layers=config.num_gnn_layers,
        linear_out_channels=config.linear_out_channels,
        esm_dim=config.esm_dim
    ).to(device)
    
    weights_path = "best_model_rmse.pt" if os.path.exists("best_model_rmse.pt") else "model_final_pKd.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print(f"--- [3/5] Анализ внимания для комплекса: {pdb_id} ---")
    with torch.no_grad():
        batch = Batch.from_data_list([data]).to(device)
        _ = model(batch) # Прогон для активации весов внимания
        
        # Извлекаем веса внимания с последнего GNN слоя
        # Мы добавили self.attn_weights в код слоя Gate_Augmented_GATv2_Layer
        last_layer = model.gnn_layers[-1]
        edge_index_attn, alpha = last_layer.attn_weights
        
        # Усредняем по головам внимания (heads)
        alpha_avg = alpha.mean(dim=-1).squeeze()

        # Накапливаем важность: сколько внимания "пришло" в каждый узел
        node_importance = torch.zeros(batch.x.size(0))
        node_importance.scatter_add_(0, edge_index_attn[1], alpha_avg)

        # Берем ТОП-10 среди ВСЕХ атомов комплекса
        top_val, top_idx = torch.topk(node_importance, 10)
        
        lig_found = []
        prot_found = []

        print("\n" + "="*60)
        print(f"ABSOLUTE TOP 10 ATOMS (RANKED BY ATTENTION)")
        print("="*60)
        
        for i, idx_raw in enumerate(top_idx.tolist()):
            score = top_val[i].item()
            
            if idx_raw < num_lig_atoms:
                # Это атом лиганда
                atom = ligand_mol.GetAtomWithIdx(idx_raw)
                symbol = atom.GetSymbol()
                print(f"{i+1}. [LIGAND]  {symbol}(idx:{idx_raw}) | Score: {score:.4f}")
                lig_found.append(idx_raw)
            else:
                # Это атом белка
                p_idx = idx_raw - num_lig_atoms
                atom = pocket_mol.GetAtomWithIdx(p_idx)
                info = atom.GetPDBResidueInfo()
                res_name = info.GetResidueName().strip() if info else "UNK"
                res_num = info.GetResidueNumber() if info else "?"
                symbol = atom.GetSymbol()
                print(f"{i+1}. [PROTEIN] {res_name}{res_num}:{symbol}(idx:{p_idx}) | Score: {score:.4f}")
                prot_found.append(p_idx)
        
        print("="*60)
        print(f"Итог: Лиганд — {len(lig_found)} ат., Белок — {len(prot_found)} ат.")

    print("\n--- [4/5] Генерация скрипта PyMOL ---")
    pml_file = f"visualize_top10_global_{pdb_id}.pml"
    with open(pml_file, "w") as f:
        f.write(f"load {pdb_id}_pocket.pdb\n")
        f.write(f"load {pdb_id}_ligand.mol2, ligand\n")
        f.write("bg_color white\n")
        f.write("show sticks, ligand\n")
        f.write("set stick_radius, 0.15\n")
        f.write("color marine, ligand\n")
        
        # Выделяем и красим топ-атомы лиганда
        if lig_found:
            lig_ids = "+".join([str(i+1) for i in lig_found])
            f.write(f"select top_lig, ligand and index {lig_ids}\n")
            f.write("show spheres, top_lig\n")
            f.write("color red, top_lig\n")
        
        # Выделяем и красим топ-атомы белка
        if prot_found:
            prot_ids = "+".join([str(i+1) for i in prot_found])
            f.write(f"select top_prot, index {prot_ids}\n")
            f.write("show spheres, top_prot\n")
            f.write("color yellow, top_prot\n")
            
        f.write("set sphere_scale, 0.35\n")
        f.write("center ligand\n")
        f.write("ray 1000, 800\n")

    print(f"--- [5/5] Готово! Скрипт сохранен: {pml_file} ---")

if __name__ == "__main__":
    extract_top_10_global_importance()