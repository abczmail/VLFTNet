import random
import os
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from models.transformer import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention, TransformerEnsemble
from visualize import visualize_grid_attention_v2
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import time

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
                # out, _ = model(mode='rl', images=images, max_len=seq_len, eos_idx=text_field.vocab.stoi['<eos>'], beam_size=beam_size, out_size=1)
                out, _ = model.beam_search(images, seq_len, text_field.vocab.stoi['<eos>'], beam_size, 1)
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

    # val model best
    # {'BLEU': [0.8216863483134279, 0.6713929488229985, 0.5253854708093316, 0.4030621255048738], 'METEOR': 0.2971739996464571, 'ROUGE': 0.5963528195946346, 'CIDEr': 1.3487934960583912}
    # parser.add_argument('--models_path', type=list, default=[
    #     './test_online/models/S2_w_2_ga_goa_relu_repeat_best_test.pth',
    #     './test_online/models/S2_w_4_ga_goa_relu_repeat_best_test.pth',
    #     './test_online/models/S2_w_1_ga_goa_relu_best_test.pth'
    # ])

    # parser.add_argument('--models_path', type=list, default=[
    #     './test_online/models/S2_w_1_ga_goa_relu_best_test.pth',
    #     './test_online/models/S2_w_2_ga_goa_relu_repeat_best_test.pth',
    #     './test_online/models/S2_w_3_ga_goa_repeat_best_test.pth',
    #     './test_online/models/S2_w_4_ga_goa_relu_repeat_best_test.pth',
    #     './test_online/models/S2_w_5_ga_goa_relu_repeat_best_test.pth',
    #     './test_online/models/S2_w_6_ga_goa_relu_repeat_best_test.pth',
    # ])

    parser.add_argument('--models_path', type=list, default=[
        './test_online/models/s2/S2_1_ablation_base_best_test.pth',
        #{'BLEU': [0.8159057437407784, 0.6653052567845859, 0.5218835409222313, 0.4022709016167793], 'METEOR': 0.2965109377778364, 'ROUGE': 0.5931760429360942, 'CIDEr': 1.3555390385007824}
        './test_online/models/s2/S2_1_ablation_base_repeat_best_test.pth',
        #{'BLEU': [0.8170759459627032, 0.666469403975871, 0.5229006192337848, 0.40329127799832315], 'METEOR': 0.2971648940252118, 'ROUGE': 0.594193120701852, 'CIDEr': 1.360648329535032}
        './test_online/models/s2/S2_2_ablation_base_repeat_best_test.pth',
        #{'BLEU': [0.8171795191229431, 0.6667407554653632, 0.5235718380942451, 0.4043092584453019], 'METEOR': 0.29719670417006017, 'ROUGE': 0.5941272154308214, 'CIDEr': 1.3590883045149884}
        './test_online/models/s2/s2_checkpoint_last.pth',
        './test_online/models/s2/S2_3_ablation_base_repeat_best_test.pth'
        #{'BLEU': [0.8166725465374309, 0.6660996879592269, 0.5227112215236179, 0.4030298360622529], 'METEOR': 0.2964951248008772, 'ROUGE': 0.5933734570942877, 'CIDEr': 1.360089931305572}
    ])

    #{'BLEU': [0.8178264756637459, 0.6679053221188139, 0.524762904941664, 0.4053123845500818],'METEOR': 0.29762389960955093, 'ROUGE': 0.5949333191419296, 'CIDEr': 1.3633633122491957}
    # parser.add_argument('--models_path', type=list, default=[
    #     './test_online/models/s2/S2_1_ablation_base_best_test.pth',
    #     './test_online/models/s2/S2_1_ablation_base_repeat_best_test.pth',
    #     './test_online/models/s2/s2_checkpoint_last.pth',
    # ])
    #{'BLEU': [0.8174109549280422, 0.6675270705512002, 0.5250248928024038, 0.4058622934257362], 'METEOR': 0.29765380225517013, 'ROUGE': 0.5949415941508484, 'CIDEr': 1.3635960835229584}
    # parser.add_argument('--models_path', type=list, default=[
    #     './test_online/models/s2/S2_1_ablation_base_best_test.pth',
    #     # './test_online/models/s2/S2_1_ablation_base_repeat_best_test.pth',
    #     './test_online/models/s2/s2_checkpoint_last.pth',
    # ])
    # {'BLEU': [0.8185056345949379, 0.6684544386147896, 0.5260458642395504, 0.40679125637625874], 'METEOR': 0.29777787476604267, 'ROUGE': 0.5954380296377878, 'CIDEr': 1.363431178536433}
    # parser.add_argument('--models_path', type=list, default=[
    #     # './test_online/models/s2/S2_1_ablation_base_best_test.pth',
    #     './test_online/models/s2/S2_1_ablation_base_repeat_best_test.pth',
    #     './test_online/models/s2/s2_checkpoint_last.pth',
    #     # './test_online/models/s2/S2_3_ablation_base_repeat_best_test.pth'
    # ])


    # test model best
    # {'BLEU': [0.8150463729711657, 0.662849281388607, 0.5185232418603183, 0.39804812585547816], 'METEOR': 0.2949237166763598, 'ROUGE': 0.5907301622926929, 'CIDEr': 1.3452945538206447}
    # parser.add_argument('--models_path', type=list, default=[
    #     './test_online/models/S2_w_6_ga_goa_relu_repeat_best.pth',
    #     './test_online/models/S2_w_3_ga_goa_repeat_best.pth'
    # ])

    # parser.add_argument('--models_path', type=list, default=[
    #     './test_online/models/S2_w_2_ga_goa_relu_repeat_best_test.pth',
    #     './test_online/models/S2_w_2_ga_goa_relu_repeat_best_test.pth',
    #     './test_online/models/S2_w_2_ga_goa_relu_repeat_best_test.pth',
    #     './test_online/models/S2_w_2_ga_goa_relu_repeat_best_test.pth',
    # ])

    args = parser.parse_args()

    print('Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=49, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, val_dataset, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open(args.vocab_path, 'rb'))

    # Model and dataloaders
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={'m': args.m})
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])

    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder, args.num_clusters, len(text_field.vocab), 54, text_field.vocab.stoi['<pad>'], 512).to(device)

    # 集成模型
    ensemble_model = TransformerEnsemble(model=model, weight_files=args.models_path)


    # data = torch.load(args.model_path)

    # model.load_state_dict({k.replace('module.',''):v for k,v in data['state_dict'].items()})
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'add_text': text_field})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'add_text':text_field})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size, num_workers=args.workers)

    # print('val')
    # scores = predict_captions(ensemble_model, dict_dataloader_val, text_field)
    # print(scores)

    # scores = predict_captions(model, dict_dataloader_test, text_field)
    print('test')
    scores = predict_captions(ensemble_model, dict_dataloader_test, text_field)
    print(scores)
    print('it costs {} s to test.'.format(time.time() - start_time))