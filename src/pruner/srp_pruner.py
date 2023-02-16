import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from decimal import Decimal
import os, copy, time, pickle, numpy as np, math
from .meta_pruner import MetaPruner
import utility
import matplotlib.pyplot as plt
from tqdm import tqdm
from fnmatch import fnmatch, fnmatchcase
from .utils import get_score_layer, pick_pruned_layer
pjoin = os.path.join
tensor2list = lambda x: x.data.cpu().numpy().tolist()
tensor2array = lambda x: x.data.cpu().numpy()
totensor = lambda x: torch.Tensor(x)

class Pruner(MetaPruner):
    def __init__(self, model, args, logger, passer):
        super(Pruner, self).__init__(model, args, logger, passer)
        loader = passer.loader
        ckp = passer.ckp
        loss = passer.loss
        self.logprint = ckp.write_log_prune # Use another log file specifically for pruning logs
        self.netprint = ckp.write_log_prune

        # ************************** variables from RCAN ************************** 
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = model
        self.loss = loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        # **************************************************************************

        # Reg related variables
        self.reg = {}
        self.delta_reg = {}
        self._init_reg()
        self.iter_update_reg_finished = {}
        self.iter_finish_pick = {}
        self.iter_stabilize_reg = math.inf
        self.hist_mag_ratio = {}
        self.w_abs = {}
        self.mag_reg_log = {}

        # init prune_state
        self.prune_state = 'update_reg'
        
        # init pruned_wg/kept_wg if they can be determined right at the begining
        if args.greg_mode in ['part'] and self.prune_state in ['update_reg']:
            self.pr, self.pruned_wg, self.kept_wg = self._get_kept_wg_L1(align_constrained=True)

    def _init_reg(self):
        for name, m in self.model.named_modules():
            if name in self.layers:
                if self.args.wg == 'weight':
                    self.reg[name] = torch.zeros_like(m.weight.data).flatten().cuda()
                else:
                    shape = m.weight.data.shape
                    self.reg[name] = torch.zeros(shape[0], shape[1]).cuda()

    def _srp_reg(self, m, name):
        if self.pr[name] == 0:
            return True
        
        pruned = self.pruned_wg[name]
        if self.args.wg == "channel":
            self.reg[name][:, pruned] += self.args.reg_granularity_prune
        elif self.args.wg == "filter":
            self.reg[name][pruned, :] += self.args.reg_granularity_prune
        elif self.args.wg == 'weight':
            self.reg[name][pruned] += self.args.reg_granularity_prune
        else:
            raise NotImplementedError

        # when all layers are pushed hard enough, stop
        return self.reg[name].max() > self.args.reg_upper_limit

    def _update_reg(self, skip=[]):
        for name, m in self.model.named_modules():
            if name in self.layers:                
                if name in self.iter_update_reg_finished.keys():
                    continue
                if name in skip:
                    continue

                # get the importance score (L1-norm in this case)
                out = get_score_layer(m, wg='filter', criterion='l1-norm')
                self.w_abs[name] = out['l1-norm']
                
                # update reg functions, two things:
                # (1) update reg of this layer (2) determine if it is time to stop update reg
                if self.args.greg_mode in ['part']:
                    finish_update_reg = self._srp_reg(m, name)
                    
                # Check prune state
                if finish_update_reg:
                    # after 'update_reg' stage, keep the reg to stabilize weight magnitude
                    self.iter_update_reg_finished[name] = self.total_iter
                    self.logprint(f"==> {self.layer_print_prefix[name]} -- Just finished 'update_reg'. Iter {self.total_iter}. pr {self.pr[name]}")

                    # check if all layers finish 'update_reg'
                    prune_state = "stabilize_reg"
                    for n, mm in self.model.named_modules():
                        if isinstance(mm, self.LEARNABLES):
                            if n not in self.iter_update_reg_finished:
                                prune_state = ''
                                break
                    if prune_state == "stabilize_reg":
                        self.prune_state = 'stabilize_reg'
                        self.iter_stabilize_reg = self.total_iter
                        self.logprint("==> All layers just finished 'update_reg', go to 'stabilize_reg'. Iter = %d" % self.total_iter)

    def _apply_reg(self):
        for name, m in self.model.named_modules():
            if name in self.layers and self.pr[name] > 0:
                reg = self.reg[name] # [N, C]
                m.weight.grad += reg.unsqueeze(-1).unsqueeze(-1) * m.weight
                bias = False if isinstance(m.bias, type(None)) else True
                if bias:
                    m.bias.grad += reg[:, 0] * m.bias

    def _resume_prune_status(self, ckpt_path):
        raise NotImplementedError

    def _save_model(self, filename):
        savepath = f'{self.ckp.dir}/model/{filename}'
        ckpt = {
            'pruned_wg': self.pruned_wg,
            'kept_wg': self.kept_wg,
            'arch': self.model.model,
            'state_dict': self.model.model.state_dict(),
        }
        torch.save(ckpt, savepath) 
        return savepath

    def prune(self):
        self.total_iter = 0
        if self.args.resume_path:
            self._resume_prune_status(self.args.resume_path)
            self.pr, self.pruned_wg, self.kept_wg = self._get_kept_wg_L1() # get pruned and kept wg from the resumed model
            self.model = self.model.train()
            self.logprint("Resume model successfully: '{}'. Iter = {}. prune_state = {}".format(
                        self.args.resume_path, self.total_iter, self.prune_state))
        

        # Freeze some layers if necessary
        for name, module in self.model.named_modules():
            if name in self.skip_layers and self.args.freeze_skip_layers:
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
                print(f'Layer {name} is a skip_layer. It is freezed now')

        epoch = 0
        while True:
            epoch += 1
            finish_prune = self.train(epoch) # there will be a break condition to get out of the infinite loop
            if finish_prune:
                return copy.deepcopy(self.model)
            self.test()

# ************************************************ The code below refers to RCAN ************************************************ #
    def train(self, epoch):
        self.loss.step()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.lr_prune # use fixed LR in pruning

        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(learning_rate))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            self.total_iter += 1

            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)

            # @mst: print
            if self.total_iter % self.args.print_interval == 0:
                self.logprint("")
                self.logprint(f"Iter {self.total_iter} [prune_state: {self.prune_state} method: {self.args.method} compare_mode: {self.args.compare_mode} greg_mode: {self.args.greg_mode}] LR: {learning_rate} " + "-"*40)


            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            
            # -- @mst: update reg factors and apply them before optimizer updates
            if self.prune_state in ['update_reg'] and self.total_iter % self.args.update_reg_interval == 0:
                self._update_reg()

            # after reg is updated, print to check
            if self.total_iter % self.args.print_interval == 0:
                self._print_reg_status()
        
            if self.args.apply_reg: # reg can also be not applied, as a baseline for comparison
                self._apply_reg()
            # --

            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()

            # @mst: exit of reg pruning loop
            if self.prune_state in ["stabilize_reg"] and self.total_iter - self.iter_stabilize_reg == self.args.stabilize_reg_interval:
                self.logprint(f"==> 'stabilize_reg' is done. Iter {self.total_iter}. About to prune and build new model. Testing...")
                self.test()

                self._prune_and_build_new_model()
                path = self._save_model('model_just_finished_prune.pt')
                self.logprint(f"==> Pruned and built a new model. Ckpt saved: '{path}'. Testing...")
                self.test()
                return True            

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        # self.optimizer.schedule() # use fixed LR in pruning

    def _print_reg_status(self):
        pr, pruned_wg, kept_wg = self._get_kept_wg_L1(align_constrained=True)
        pr_list = []

        self.logprint('************* Regularization Status *************')
        cnt = 0
        for name, m in self.model.named_modules():
            if name in self.layers and self.pr[name] > 0:
                cnt += 1
                logstr = [self.layer_print_prefix[name]]
                logstr += [f"reg_status: min {self.reg[name].min():.5f} ave {self.reg[name].mean():.5f} max {self.reg[name].max():.5f}"]
                
                pruned, kept = pruned_wg[name], kept_wg[name]
                out = get_score_layer(m, wg='filter', criterion='l1-norm')
                w_abs = out['l1-norm']
                avg_mag_pruned, avg_mag_kept = np.mean(w_abs[pruned]), np.mean(w_abs[kept])
                
                logstr += ["average w_mag: pruned %.6f kept %.6f" % (avg_mag_pruned, avg_mag_kept)]
                logstr += [f'Iter {self.total_iter}']
                logstr += [f'cstn' if name in self.constrained_layers else 'free']
                logstr += [f'pr {pr[name]}']
                self.logprint(' | '.join(logstr))
                pr_list += [pr[name]]

    def test(self):
        is_train = self.model.training
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.model.eval()

        test_log = torch.zeros(1, len(self.loader_test), len(self.scale))
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    test_log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                test_log[-1, idx_data, idx_scale] /= len(d)
                best = test_log.max(0)
                logstr = '[{} x{}]\tTestPSNR: {:.4f} Epoch {} BestTestPSNR {:.4f} BestTestPSNREpoch {} [prune_state: {} method: {} compare_mode: {} greg_mode: {}]'.format(
                        d.dataset.name,
                        scale,
                        test_log[-1, idx_data, idx_scale],
                        epoch,
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1,
                        self.prune_state, 
                        self.args.method,
                        self.args.compare_mode,
                        self.args.greg_mode,
                    )
                # self.ckp.write_log(logstr)
                self.logprint(logstr)

        if self.args.save_results:
            self.ckp.end_background()

        torch.set_grad_enabled(True)

        if is_train:
            self.model.train()

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def _log_down_mag_reg(self, mag, name):
        step = self.total_iter
        reg = self.reg[name].max().item()
        mag = np.array(mag)
        if name not in self.mag_reg_log:
            values = [[step, reg, mag]]
            log = {
                'name': name,
                'layer_index': self.layers[name].layer_index,
                'shape': self.layers[name].size,
                'values': values,
            }
            self.mag_reg_log[name] = log
        else:
            values = self.mag_reg_log[name]['values']
            values.append([step, reg, mag])