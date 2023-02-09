import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lrate = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}] Learning rate: {}'.format(epoch, lrate)
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.2f}+{:.2f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        return lrate

    def test(self, lrate):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(self.ckp.metric_print_prefix + 
                    '[{} x{}] Epoch {} TestPSNR {:.4f} Last10AvgTestPSNR {:.4f} BestTestPSNR {:.4f} BestTestPSNREpoch {} LR {}'.format(
                        d.dataset.name,
                        scale,
                        epoch,
                        self.ckp.log[-1, idx_data, idx_scale],
                        self.ckp.log[-10:, idx_data, idx_scale].mean().item(),
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1,
                        lrate,
                    ), refresh=True
                )

        # self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        # self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        # Save checkpoint
        if not self.args.test_only:
            self.ckp.save(
                self, 
                epoch, 
                is_best=(best[1][0, 0] + 1 == epoch),
                lrate=lrate,
            )
        if self.args.only_keep_minlrate_models:
            ckpt_dir = self.ckp.get_path('model')
            ckpts = [os.path.join(ckpt_dir, x) for x in os.listdir(ckpt_dir) if x.endswith('.pt') and 'lrate' in x]
            if len(ckpts):
                lrates = []
                for ck in ckpts:
                    lrate_ = float(ck.split('lrate')[1].split('.pt')[0])
                    lrates += [lrate_]
                for ck in ckpts:
                    lrate_ = float(ck.split('lrate')[1].split('.pt')[0])
                    if lrate_ != min(lrates):
                        os.remove(ck)
                        print(f'==> Remove {ck} because of --only_keep_minlrate_models')

        # self.ckp.write_log(
        #     'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        # )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test(lrate=-1)
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

