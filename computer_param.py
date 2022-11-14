import random
import os
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from models.transformer import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention
from visualize import visualize_grid_attention_v2
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import time

from thop import profile
from ptflops import get_model_complexity_info
from torchstat import stat
from torchvision import models
from thop import profile
from thop import clever_format

from torchsummaryX import summary

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    seq_len = 20
    beam_size = 5
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt, _) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model(mode='rl', images=images, max_len=seq_len, eos_idx=text_field.vocab.stoi['<eos>'],
                               beam_size=beam_size, out_size=1)
                # print(out.size(), att_map.size())
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


if __name__ == '__main__':
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--m', type=int, default=40)

    parser.add_argument('--features_path', type=str, default='./X101-features/X101_grid_feats_coco_trainval.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='./m2_annotations')

    # the path of tested model and vocabulary
    parser.add_argument('--model_path', type=str, default='saved_transformer_models/demo_rl_v5_best_test.pth')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl')
    parser.add_argument('--num_clusters', type=int, default=5)
    args = parser.parse_args()

    print('Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=49, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open(args.vocab_path, 'rb'))

    # Model and dataloaders
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention,
                                 attention_module_kwargs={'m': args.m})
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])

    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder, args.num_clusters, len(text_field.vocab), 54, text_field.vocab.stoi['<pad>'], 512).to(device)


    data = torch.load(args.model_path)

    model.load_state_dict({k.replace('module.', ''): v for k, v in data['state_dict'].items()})
    # model.load_state_dict(data['state_dict'], strict=False)
    #
    # dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'add_text': text_field})
    # dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    # total_params += sum(p.numel() for p in model.buffers())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params / (1024 * 1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')

    print("++++++++++++++++++++++++++++++")
    from thop import profile

    input = torch.randn(1, 50, 49, 2048)
    flops, params = profile(model, inputs=(input,))
    print(flops)
    print(params)




