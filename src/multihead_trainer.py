import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os, copy, datetime
import pandas as pd
from tqdm import tqdm
from torchvision import transforms

torch_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from metric_utils import HardNegativePairSelector, RandomNegativeTripletSelector
from metrics import MetricsCollection
from pillid_datasets import BalancedBatchSamplerPillID, SingleImgPillID
from sanitytest_eval import create_eval_dataloaders
from models.multihead_model import MultiheadModel
from models.embedding_model import EmbeddingModel
from models.losses import MultiheadLoss
from metric_test_eval import MetricEmbeddingEvaluator

def train(ref_only_df, cons_train_df, cons_val_df, label_encoder, transform, labelcol, batch_size, _, args, n_epochs, *pos_args, **kwargs):
    results_dir = os.path.join(args.results_dir, "latest_run")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'train_metrics.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    E_model = EmbeddingModel(network=args.appearance_network, pooling=args.pooling, dropout_p=args.dropout, cont_dims=args.metric_embedding_dim, pretrained=True)
    model = MultiheadModel(E_model, len(label_encoder.classes_), train_with_side_labels=True).to(device)
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(SingleImgPillID(pd.concat([ref_only_df, cons_train_df]), label_encoder, train=True, transform=transform, labelcol=labelcol), 
                                            batch_sampler=BalancedBatchSamplerPillID(pd.concat([ref_only_df, cons_train_df]), batch_size, labelcol), num_workers=2),
        'val': torch.utils.data.DataLoader(SingleImgPillID(pd.concat([ref_only_df, cons_val_df]), label_encoder, train=False, transform=transform, labelcol=labelcol), 
                                          batch_sampler=BalancedBatchSamplerPillID(pd.concat([ref_only_df, cons_val_df]), batch_size, labelcol), num_workers=2),
        'eval': create_eval_dataloaders(cons_val_df, label_encoder, transform, labelcol, 24)[0],
        'ref': create_eval_dataloaders(ref_only_df, label_encoder, transform, labelcol, 24)[0]
    }

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.lr_patience)
    criterion = MultiheadLoss(len(label_encoder.classes_), args.metric_margin, HardNegativePairSelector(), args.metric_margin, RandomNegativeTripletSelector(args.metric_margin), use_cosine=True)

    epoch_metrics = MetricsCollection(); best_ap = -1.0
    for epoch in range(n_epochs):
        print(f'\n========== EPOCH {epoch}/{n_epochs-1} ==========')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            pbar = tqdm(dataloaders[phase], desc=phase, ncols=100)
            for batch in pbar:
                inputs, labels = batch['image'].to(device), batch['label'].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(inputs, labels)
                    loss_out = criterion(out, labels, is_front=batch['is_front'], is_ref=batch['is_ref'])
                    if phase == 'train': loss_out['loss'].backward(); optimizer.step()
                epoch_metrics.add(phase, 'loss', loss_out['loss'].item(), inputs.size(0))
            
            if phase == 'val':
                res, _ = MetricEmbeddingEvaluator(model, metric_evaluator_type='cosine').eval_model(device, dataloaders)
                ap = res.get('micro-ap', 0)
                epoch_metrics.add('val', 'micro-ap', ap, 1)
                
                now_str = datetime.datetime.now().strftime('%H:%M:%S')
                
                # 1. ÉP LƯU CHECKPOINT (XÓA FILE CŨ TRƯỚC)
                ckpt_path = os.path.join(results_dir, 'last_checkpoint.pth')
                if os.path.exists(ckpt_path): os.remove(ckpt_path) # <--- Sát thủ diệt UI lag
                torch.save(model.state_dict(), ckpt_path)
                os.sync()
                print(f"💾 [{now_str}] Đã tạo mới: last_checkpoint.pth")
                
                # 2. ÉP LƯU BEST MODEL
                if ap > best_ap:
                    best_ap = ap
                    best_path = os.path.join(results_dir, 'best_model.pth')
                    if os.path.exists(best_path): os.remove(best_path)
                    torch.save(model.state_dict(), best_path)
                    os.sync()
                    print(f"🏆 [{now_str}] KỶ LỤC MỚI: best_model.pth (AP: {ap:.4f})")
                
                # 3. LƯU CSV
                try:
                    log_data = {'epoch': epoch}
                    source = epoch_metrics.metrics if hasattr(epoch_metrics, 'metrics') else epoch_metrics
                    for p in ['train', 'val']:
                        if p in source:
                            for k, m in source[p].items():
                                if len(m.history) > 0: log_data[f"{p}_{k}"] = m.history[-1]
                    df_log = pd.DataFrame([log_data])
                    is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) < 10
                    df_log.to_csv(csv_path, mode='a', header=is_new, index=False)
                    os.sync()
                except Exception as e:
                    print(f"⚠️ LỖI LƯU CSV: {e}")
                    
        val_loss_hist = epoch_metrics.metrics['val']['loss'].history if hasattr(epoch_metrics, 'metrics') else epoch_metrics['val']['loss'].history
        scheduler.step(val_loss_hist[-1])
        
    return model, {}
