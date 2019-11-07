#

# rewrite for it with pytorch

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F
import sys, os
import pickle
import argparse
import time

# =====
# general confs

np.random.seed(1337)
FONT_FILE = "DejaVuSans.ttf"
IMAGE_SIZE = (15, 60)
NUMBER_MAX_DIGIT = 7
ADD_MAXV = 4999999
MUL_MAXV = 3160
DATA_SIZES = [150000, 30000, 30000]  # train/dev/test
# model
BATCH_SIZE = 256
MAX_EPOCH = 100

# =====
# data part

def printing(x):
    print(x, file=sys.stderr, flush=True)

def show_image(arr, sarr=None):
    # input is [x, y, 2]
    t_arr = np.transpose(arr, [2,0,1]).reshape([-1, arr.shape[1]])
    if sarr is not None:
        t_sarr = np.transpose(sarr, [2,0,1]).reshape([-1, sarr.shape[1]])
        ax1 = plt.subplot(2, 1, 1)
        ax1.imshow(t_arr, cmap='gray')
        ax1.axis('off')
        ax2 = plt.subplot(2, 1, 2)
        ax2.imshow(t_sarr, cmap='gray')
        ax2.axis('off')
    else:
        plt.imshow(show_arr, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class ArithmeticDataSet:
    FONT = ImageFont.truetype(FONT_FILE, 12)

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    # number, output_size -> np.arr
    @staticmethod
    def get_img(n: int, img_sz):
        im = np.zeros(img_sz)
        pil_im = Image.fromarray(im)
        draw = ImageDraw.Draw(pil_im)
        font = ArithmeticDataSet.FONT
        sz = font.getsize(str(n))
        draw.text((img_sz[1] - sz[0], 2), str(n), font=font)
        im = np.asarray(pil_im)
        return im

    # generate the data
    @staticmethod
    def get_dataset(num_points: List[int], oper: str, img_sz):
        # set max-sample value
        max_v = {"ADD": ADD_MAXV, "MUL": MUL_MAXV}[oper]
        oper_f = {"ADD": np.sum, "MUL": np.prod}[oper]
        # get randoms (no-repeat)
        num_all = np.sum(num_points)
        n1s = np.random.randint(0, max_v, 2*num_all)
        n2s = np.random.randint(0, max_v, 2*num_all)
        hit_set = set()
        for a,b in zip(n1s, n2s):
            k = (int(a), int(b))
            if k not in hit_set:
                hit_set.add(k)
            if len(hit_set)>=num_all:
                break
        assert len(hit_set)>=num_all, "Not enough data!!"
        # get all the data and split
        all_pairs = list(hit_set)
        np.random.shuffle(all_pairs)  # shuffle the data
        all_datasets = []
        cur_base = 0
        for cur_count in num_points:
            # get input
            cur_pairs = all_pairs[cur_base:cur_base+cur_count]
            assert len(cur_pairs) == cur_count
            cur_base += cur_count
            cur_pair_arr = np.asarray(cur_pairs)  # [N, 2]
            # get result
            cur_result_arr = oper_f(cur_pair_arr, -1)  # [N, ]
            # get output digits
            cur_result_digits = []
            tmp_result_arr = np.copy(cur_result_arr)
            for d in range(NUMBER_MAX_DIGIT):
                cur_result_digits.append(tmp_result_arr%10)
                tmp_result_arr = tmp_result_arr // 10
            assert np.allclose(tmp_result_arr, 0)
            cur_result_digits_arr = np.stack(cur_result_digits, -1)  # [N, Digit] low2high
            # get input images
            cur_input_arr = np.zeros((cur_count, img_sz[0], img_sz[1], 2))
            for this_idx, this_pair in enumerate(cur_pairs):
                n1, n2 = this_pair
                cur_input_arr[this_idx, :, :, 0] = ArithmeticDataSet.get_img(n1, img_sz)
                cur_input_arr[this_idx, :, :, 1] = ArithmeticDataSet.get_img(n2, img_sz)
                if (this_idx+1) % 1000 == 0:
                    printing(f"Getting image: {this_idx}/{cur_count}")
            cur_dataset = ArithmeticDataSet(input=cur_input_arr, pair=cur_pair_arr,
                                            result=cur_result_arr, digits=cur_result_digits_arr)
            all_datasets.append(cur_dataset)
        return all_datasets

# =====
# model

class ArithmeticModel(Module):
    def __init__(self, img_sz, hsize: int, sep: int):
        super().__init__()
        # encoder
        H = hsize
        self.hidden_size = H
        self.sep = sep
        if sep:
            self.enc = self._get_enc(img_sz, H)
            self.enc2 = self._get_enc(img_sz, H)
            self.enc3 = self._get_enc(img_sz, H)
            self.enc4 = self._get_enc(img_sz, H)
            self.pred = torch.nn.Linear(H, (NUMBER_MAX_DIGIT-3)*10)
            self.pred2 = torch.nn.Linear(H, 10)
            self.pred3 = torch.nn.Linear(H, 10)
            self.pred4 = torch.nn.Linear(H, 10)
        else:
            self.enc = self._get_enc(img_sz, H)
            self.pred = torch.nn.Linear(H, NUMBER_MAX_DIGIT*10)

    def _get_enc(self, img_sz, H):
        return torch.nn.Sequential(
            torch.nn.Linear(np.prod(img_sz)*2, H),  # flatten the input
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU())

    # [BS, *] -> [BS, Digit, 10]
    def _pred(self, input_t):
        if self.sep:
            enc_t = self.enc(input_t)  # [BS, H]
            pred_t = self.pred(enc_t).view([-1, NUMBER_MAX_DIGIT-3, 10])  # [BS, Digit-3, 10]
            enc_t2 = self.enc2(input_t)
            enc_t3 = self.enc3(input_t)
            enc_t4 = self.enc4(input_t)
            pred_t2 = self.pred2(enc_t2).view([-1, 1, 10])  # [BS, 1, 10]
            pred_t3 = self.pred3(enc_t3).view([-1, 1, 10])  # [BS, 1, 10]
            pred_t4 = self.pred4(enc_t4).view([-1, 1, 10])  # [BS, 1, 10]
            ret_t = torch.cat([pred_t[:, :2], pred_t2, pred_t3, pred_t4, pred_t[:, -2:]], -2)  # [BS, Digit, 10]
        else:
            enc_t = self.enc(input_t)  # [BS, H]
            ret_t = self.pred(enc_t).view([-1, NUMBER_MAX_DIGIT, 10])  # [BS, Digit, 10]
        return ret_t

    # [N, *], [N, Digit](optional)
    def run(self, input_arr, digits_arr=None):
        bsize = input_arr.shape[0]
        input_t = torch.tensor(input_arr, dtype=torch.float32).view([bsize, -1])  # flatten: [BS, *]
        pred_t = self._pred(input_t)
        all_logprobs_t = F.log_softmax(pred_t, -1)
        # the target: [BS, Digit]
        if digits_arr is None:
            ret_logprobs, ret_digits = all_logprobs_t.max(-1)
        else:
            ret_digits = torch.tensor(digits_arr, dtype=torch.long)
            ret_logprobs = all_logprobs_t.gather(-1, ret_digits.unsqueeze(-1)).squeeze(-1)
        return ret_digits, ret_logprobs

    # [*], [Digit], int
    def salience(self, one_input_arr, which_digit, what_digit):
        input_t = torch.tensor(one_input_arr, dtype=torch.float32)
        input_t.requires_grad = True
        pred_t = self._pred(input_t.view([1, -1]))  # [BS, Digit, 10]
        all_logprobs_t = F.log_softmax(pred_t, -1)  # [BS, Digit, 10]
        one_loss = - all_logprobs_t[0][which_digit][what_digit]
        one_loss.backward()
        return input_t.grad.abs().cpu().numpy()  # take the abs value

# =====
# running

def iter_data(input_arr, digits_arr, batch_size):
    all_size = len(input_arr)
    assert len(digits_arr) == all_size
    cur_base = 0
    while cur_base < all_size:
        slice_input = input_arr[cur_base:cur_base+batch_size]
        slice_digit = digits_arr[cur_base:cur_base+batch_size]
        yield slice_input, slice_digit
        cur_base += batch_size

#
def forever_iter_data(*args):
    while True:
        for x in iter_data(*args):
            yield x

def do_test(model, test_input, test_digits, batch_size, test_name: str):
    all_corr, all_logprobs = [], []
    for slice_input, slice_digit in iter_data(test_input, test_digits, batch_size):
        # first get force-loss
        _, logprobs = model.run(slice_input, slice_digit)
        # then predict
        pred_digits, _ = model.run(slice_input, None)
        # collect
        corr_matrix = (pred_digits.detach().cpu().numpy() == slice_digit).astype(np.int)
        all_logprobs.append(logprobs.detach().cpu().numpy())
        all_corr.append(corr_matrix)
    all_corr = np.concatenate(all_corr, 0)
    all_logprobs = np.concatenate(all_logprobs, 0)
    # =====
    avg_logprob = all_logprobs.mean()
    acc_inst = all_corr.prod(-1).mean()
    acc_digits = all_corr.mean()
    acc_each_digit = all_corr.mean(0)
    acc_each_digit_s = [int(z*10000)/10000 for z in acc_each_digit]
    ss = f"Testing on {test_name}: logprob={avg_logprob:.4f}, acc={acc_inst:.4f}|{acc_digits:.4f}||{acc_each_digit_s}"
    printing(ss)
    return avg_logprob, acc_inst, acc_digits

def main():
    # =====
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--oper", type=str, required=True)
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--model", type=str, default="model")
    parser.add_argument("--hsize", type=int, default=256)
    parser.add_argument("--sep", type=int, default=0)
    parser.add_argument("--train_instances", type=int, default=0)  # 0 means all
    args = parser.parse_args()
    printing(f"RUN with {args}")
    mode, oper, data_prefix, model_prefix = args.mode, args.oper, args.data, args.model
    hsize, sep = args.hsize, args.sep
    # =====
    # get data
    data_file = ".".join([data_prefix, oper, "pic"])
    if os.path.exists(data_file):
        with open(data_file, "rb") as fd:
            data_train, data_dev, data_test = pickle.load(fd)
            printing(f"Loading data from {data_file}: train={len(data_train.input)}, "
                     f"dev={len(data_dev.input)}, test={len(data_test.input)}")
    else:
        printing(f"Build data and save to {data_file}: {DATA_SIZES}")
        data_train, data_dev, data_test = ArithmeticDataSet.get_dataset(DATA_SIZES, oper, IMAGE_SIZE)
        with open(data_file, "wb") as fd:
            pickle.dump([data_train, data_dev, data_test], fd)
    # =====
    # get model
    model = ArithmeticModel(img_sz=IMAGE_SIZE, hsize=hsize, sep=sep)
    model_path = f"{model_prefix}.{oper}.pt"
    if mode!="train":
        printing(f"Load model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        printing("Build model and train.")
    # =====
    # go
    input_train, input_dev, input_test = [z.input for z in [data_train, data_dev, data_test]]
    digits_train, digits_dev, digits_test = [z.digits for z in [data_train, data_dev, data_test]]
    # -----
    mu = np.mean(input_train, axis=0)
    input_train -= mu
    input_dev -= mu
    input_test -= mu
    # -----
    if mode=="train":
        train_instances = args.train_instances
        if train_instances == 0:
            train_instances = len(input_train)
        epoch_num_instance = len(input_train)
        input_train, digits_train = input_train[:train_instances], digits_train[:train_instances]
        # training epochs
        dev_best_acc = -1.
        dev_best_record = None
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # =====
        data_iter = forever_iter_data(input_train, digits_train, BATCH_SIZE)
        for eidx in range(MAX_EPOCH):
            printing(f"Starting Epoch {eidx} at {time.ctime()}")
            # train
            model.train()
            all_losses = []
            cur_num_instance = 0
            while True:
                slice_input, slice_digit = next(data_iter)
                optimizer.zero_grad()
                _, logprobs = model.run(slice_input, slice_digit)
                loss = - logprobs.mean()
                all_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                # end one epoch
                cur_num_instance += len(slice_input)
                if cur_num_instance >= epoch_num_instance:
                    break
            printing(f"Current Epoch's avg loss is {np.average(all_losses)}")
            # dev/test
            model.eval()
            dev_avg_logprob, dev_acc_inst, dev_acc_digits = do_test(model, input_dev, digits_dev, BATCH_SIZE, "DevSet")
            test_avg_logprob, test_acc_inst, test_acc_digits = do_test(model, input_test, digits_test, BATCH_SIZE, "TestSet")
            if dev_acc_digits > dev_best_acc:
                dev_best_acc = dev_acc_digits
                dev_best_record = (eidx, dev_avg_logprob, dev_acc_inst, dev_acc_digits,
                                   test_avg_logprob, test_acc_inst, test_acc_digits)
                printing(f"Get best on dev, save model, best record is {dev_best_record}")
                torch.save(model.state_dict(), model_path)
        printing(f"Finish training, best record is {dev_best_record}")
    elif mode=="test":
        model.eval()
        do_test(model, input_test, digits_test, BATCH_SIZE, "TestSet")
    elif mode=="salience":
        model.train()
        while True:
            line = input(">> ")  # enter two numbers and which digit to look at (from 0 to 6)
            n1, n2, which_digit = [int(z) for z in line.split()]
            cur_input_arr = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 2))
            cur_input_arr[:, :, 0] = ArithmeticDataSet.get_img(n1, IMAGE_SIZE)
            cur_input_arr[:, :, 1] = ArithmeticDataSet.get_img(n2, IMAGE_SIZE)
            res = (n1+n2) if oper=="ADD" else (n1*n2)
            what_digit = (res // (10**which_digit)) % 10
            printing(f"Currently looking at {n1} {oper} {n2} = {res}, digit[{which_digit}]={what_digit}")
            cur_saliency_arr = model.salience(cur_input_arr, which_digit, what_digit)
            show_image(cur_input_arr, cur_saliency_arr)
    else:
        raise NotImplementedError()

# python3 ~ --mode train --oper ADD
# python3 zrun.py --mode salience --oper MUL
if __name__ == '__main__':
    main()

# running with different training sizes
"""
# bash _run_ts.sh |& tee _log_ts
for oper in ADD MUL; do
for ts in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 150000; do
echo "ZRUN ${oper} ${ts}"
python3 zrun.py --mode train --oper $oper --train_instances $ts
done
done
"""
# running with different hsizes for MUL
"""
# bash _run_hs.sh |& tee _log_hs
for oper in ADD MUL; do
for hs in 64 128 256 512 1024; do
echo "ZRUN ${oper} ${hs}"
python3 zrun.py --mode train --oper $oper --hsize $hs
done
done
"""
# running with sep mode
"""
# bash _run_sep.sh |& tee _log_sep
for oper in MUL; do
for sep in 1 0; do
for hs in 64 128 256 512 1024; do
echo "ZRUN ${oper} ${hs} ${sep}"
python3 zrun.py --mode train --oper $oper --hsize $hs --sep $sep
done
done
done
"""
# running with sep mode and with different sizes
"""
# bash _run_ts2.sh |& tee _log_ts2
for oper in MUL; do
for ts in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 150000; do
echo "ZRUN-sep ${ts}"
python3 zrun.py --mode train --oper $oper --train_instances $ts --sep 1
done
done
"""
