import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F

#from model import Model
from data.utils import normalize, normalize_reverse

from model.MMPRNN import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Deblurring')

    # experiment mark
    parser.add_argument('--description', type=str, default='develop', help='experiment description')

    # data parameters
    parser.add_argument('--data_format', type=str, default='RGB', help='RGB or RAW')

    # model parameters
    parser.add_argument('--n_features', type=int, default=18, help='base # of channels for Conv')
    parser.add_argument('--n_blocks', type=int, default=15, help='# of blocks in middle part of the model')
    parser.add_argument('--n_blocks_a', type=int, default=9, help='# of blocks in middle part of the model')
    parser.add_argument('--n_blocks_b', type=int, default=10, help='# of blocks in middle part of the model')
    parser.add_argument('--future_frames', type=int, default=2, help='use # of future frames')
    parser.add_argument('--past_frames', type=int, default=2, help='use # of past frames')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--weight', type=str, default='mmp-rnn-gopro2.pth')
    args, _ = parser.parse_known_args()
    
    #args.model = 'MMPRNN'
    args.do_skip = True
    args.normalize = True
    args.centralize = True
    args.dataset = 'gopro'
    
    model = Model(args).cuda()
    # model_st = model.state_dict()
    # checkpt_st = torch.load(args.weight, map_location=lambda storage, loc: storage.cuda())
    # for key, key1 in zip(checkpt_st.keys(), model_st.keys()):
    #     model_st[key1] = checkpt_st[key]
    # torch.save(model_st, 'mmp-rnn-gopro2.pth')
    model.load_state_dict(torch.load(args.weight, map_location=lambda storage, loc: storage.cuda()))
    model.eval()
    vc = cv2.VideoCapture(args.input)
    if not vc.isOpened():
        print('Open file {} failed!'.format(input))
        exit()
    fps = vc.get(cv2.CAP_PROP_FPS)
    total_frame = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    ds_ratio = 4
    s_height = int(height / ds_ratio)
    s_width = int(width / ds_ratio)
    device = torch.device('cuda')
    s = torch.zeros(1, args.n_features, s_height, s_width).to(device)
    mid = torch.zeros(1, 2*args.n_features, s_height, s_width).to(device)
    hs0 = torch.zeros(1, 90, s_height, s_width).to(device)
    hs = [hs0, hs0]
    input_list = []
    eof = False
    finish = False
    last = 2
    last_frame = None
    with torch.no_grad():
        with tqdm(total=int(total_frame)) as pbar:
            while not finish:
                if not eof:
                    rst, frame = vc.read()
                    if not rst:
                        eof = True
                        frame = last_frame
                else:
                    last -= 1
                    if last == 0:
                        finish = True
                        break
                    frame = last_frame
                input_tensor = normalize(torch.from_numpy(frame.transpose(2, 0, 1)).float().cuda(), True, True)
                input_tensor = torch.unsqueeze(input_tensor, 0)
                input_list.append(input_tensor)

                output_tensor, hs, s, mid = model(input_list, hs, s, mid)
                if output_tensor is not None:
                    input_list.pop(0)
                    hs.pop(0)
                    deblur_img = normalize_reverse(output_tensor, True, True)
                    deblur_img = deblur_img.detach().cpu().numpy().transpose((1, 2, 0)) 
                    deblur_img = np.clip(deblur_img, 0, 255).astype(np.uint8)
                    #cv2.imwrite('test.png', deblur_img)
                    vw.write(deblur_img)
                last_frame = frame
                pbar.update(1)

                
