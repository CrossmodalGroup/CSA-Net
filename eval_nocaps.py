import os
import json
import hydra
import random
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from datasets.caption.nocaps import NocapsDataset, NoCapsCollator

from datasets.caption.field import TextField
from datasets.caption.vocab import Vocab
from models.caption import Transformer, CascadeSemanticAlignmentNetwork, CaptionGenerator

from models.caption.detector import build_detector
from models.common.attention import MemoryAttention

import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from engine.caption_engine import *
os.environ["CUDA_VISIBLE_DEVICES"] = '5' #

def main(gpu, config):
    # dist init
    torch.backends.cudnn.enabled = False
    rank = config.exp.rank * config.exp.ngpus_per_node + gpu
    dist.init_process_group('nccl', 'env://', rank=rank, world_size=config.exp.world_size)

    torch.manual_seed(config.exp.seed)
    np.random.seed(config.exp.seed)
    random.seed(config.exp.seed)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)

    # extract features
    detector = build_detector(config).to(device)

    csa_net = CascadeSemanticAlignmentNetwork(
        pad_idx=config.model.pad_idx,
        d_in=config.model.csa_feat_dim,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        attention_module=MemoryAttention,
        **config.model.csa_net,
    )
    cap_generator = CaptionGenerator(
        vocab_size=config.model.vocab_size,
        max_len=config.model.max_len,
        pad_idx=config.model.pad_idx,
        cfg=config.model.cap_generator,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        **config.model.cap_generator,
    )
    model = Transformer(
        csa_net,
        cap_generator,
        detector=detector,
        use_rel_feat=config.model.use_rel_feat,
        use_reg_feat=config.model.use_reg_feat,
        config=config,
    )
    model = model.to(device)

    if os.path.exists(config.exp.checkpoint):
        checkpoint = torch.load(config.exp.checkpoint, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("model missing:", len(missing))
        print("model unexpected:", len(unexpected))

    model = DDP(model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
    # evaluate only
    model.module.cached_features = False

    data_path = './coco_caption/nocaps'
    ann_path = os.path.join(data_path, './coco_caption/nocaps_val_image_info.json')
    root_path = os.path.join(data_path, 'val')

    vocab = Vocab(vocab_path=config.dataset.vocab_path)
    dataset = NocapsDataset(
        vocab=vocab,
        ann_path=ann_path,
        root=root_path,
        pad_idx=3,
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=NoCapsCollator(device))
    split = 'val'

    text_field = TextField(vocab_path=config.dataset.vocab_path)

    if rank == 0:
        model.eval()

        counter = 0
        times = []
        with tqdm(desc=f'Evaluation on Nocaps', unit='it', total=len(dataloader)) as pbar:

            results = []
            for it, batch in enumerate(iter(dataloader)):
                counter += 1
                start_it = time.time()
                with torch.no_grad():
                    out, _ = model(
                        batch['samples'],
                        seq=None,
                        use_beam_search=True,
                        max_len=config.model.beam_len,
                        eos_idx=config.model.eos_idx,
                        beam_size=config.model.beam_size,
                        out_size=1,
                        return_probs=False,
                    )
                torch.cuda.synchronize()
                end_it = time.time()
                times.append(end_it - start_it)

                if 'samples' in batch:
                    bs = batch['samples'].tensors.shape[0]
                elif 'rel_feat' in batch:
                    bs = batch['rel_feat'].shape[0]
                if it % 100 == 0:
                    print(
                        f"Number of iterations: {counter}, batch_size={bs}, Total time per 1 batch: {sum(times)/counter:0.5f}s"
                    )

                caps_gen = text_field.decode(out, join_words=False)
                for i, gen_i in enumerate(caps_gen):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    res = {'image_id': batch['image_id'][i], 'caption': gen_i}
                    results.append(res)
                pbar.update()

        print("Number of images:", len(results))
        with open(f'nocaps_{split}_4ds.json', 'w') as f:
            json.dump(results, f)



@hydra.main(config_path="configs/caption", config_name="eval_nocaps_config")
def run_main(config: DictConfig) -> None:
    mp.spawn(main, nprocs=config.exp.ngpus_per_node, args=(config,))


if __name__ == "__main__":
    os.environ["DATA_ROOT"] = "./coco_caption"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6688"
    run_main()