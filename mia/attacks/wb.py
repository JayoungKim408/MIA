import time
import torch
import numpy as np
from sklearn import metrics
from torch.autograd import Variable

from mia.attacks.utils import compute_distance
from mia.utils.LBFGS_pytorch import *
from mia.attacks.base import BaseAttacker


class WhiteBox(torch.nn.Module):
    def __init__(self, model, meta, z_dim, if_norm_reg=True, lambda3=0.001, batch_size=2000):
        super(WhiteBox, self).__init__()
        self.dataset = model.dataset_name
        self.model = model
        self.z_dim = z_dim
        self.meta = meta
        
        # Initialize a Transformer 
        self.transformer = self.model.transformer

        # no need to initialize lambda2 and L_lpips since it is non-image data
        self.lambda3 = lambda3
        self.batch_size = batch_size
        
        print('Use distance: l2')
        self.loss_l2_fn = compute_distance
        self.if_norm_reg = if_norm_reg

    def forward(self, z, x_gt):
        if z.shape[0] < self.batch_size:
            z_ = z
        else:
            z_ = z.reshape(self.batch_size, -1)
        self.x_hat = torch.from_numpy(self.model.sample_pbb(z_.cpu())).to('cuda')
        self.loss_l2 = self.loss_l2_fn(self.model.sample_pbb(z_.cpu()), x_gt.cpu().numpy(), self.meta)
        self.vec_loss = torch.from_numpy(self.loss_l2).to('cuda')

        if self.if_norm_reg:
            z_ = z.view(-1, self.z_dim)
            norm = torch.sum(z_ ** 2, dim=1)
            norm_penalty = (norm - self.z_dim) ** 2
            self.vec_loss += self.lambda3*norm_penalty

        return torch.mean(self.vec_loss)


class LatentZ(torch.nn.Module):
    def __init__(self, init_val):
        super(LatentZ, self).__init__()
        self.z = torch.nn.Parameter(init_val.data)

    def forward(self):
        return self.z

    def reinit(self, init_val):
        self.z = torch.nn.Parameter(init_val.data)


class WhiteBoxAttacker(BaseAttacker):

    def optimize(self, loss_model, init_val, query_data, max_func, lbfgs_lr=0.0001):
        ### store results
        all_loss = []

        ### run the optimization for all query data
        plus = 1 if len(query_data) % loss_model.batch_size != 0  else 0 
        for i in range(len(query_data) // loss_model.batch_size + plus):
            try:
                x_batch = query_data[i * loss_model.batch_size:(i + 1) * loss_model.batch_size]
                if x_batch.shape[0] < loss_model.batch_size:
                    x_batch = np.reshape(x_batch, [x_batch.shape[0], -1])
                else:
                    x_batch = np.reshape(x_batch, [loss_model.batch_size, -1])
                x_gt = torch.from_numpy(x_batch).to('cuda')  

                ### initialize z
                z = Variable(init_val[i * loss_model.batch_size:(i + 1) * loss_model.batch_size]).to('cuda')
                z_model = LatentZ(z)

                ### LBFGS optimizer
                optimizer = FullBatchLBFGS(z_model.parameters(), lr=lbfgs_lr, history_size=20, line_search='Wolfe',
                                        debug=False)

                ### optimize
                loss_progress = []

                def closure():
                    optimizer.zero_grad()
                    vec_loss = loss_model.forward(z_model.forward(), x_gt)
                    vec_loss_np = np.sqrt(vec_loss.detach().cpu().numpy())
                    loss_progress.append(vec_loss_np)
                    final_loss = torch.mean(vec_loss)
                    return final_loss

                for step in range(max_func):
                    loss_model.forward(z_model.forward(), x_gt)
                    final_loss = closure()
                    final_loss.backward()

                    options = {'closure': closure, 'current_loss': final_loss, 'max_ls': 20}
                    optimizer.step(options)

                    if step == 0:
                        vec_loss_curr = loss_model.vec_loss.data.cpu().numpy()

                    if step == max_func - 1:
                        vec_loss_curr = loss_model.vec_loss.data.cpu().numpy()

                        all_loss.append(vec_loss_curr)

            except KeyboardInterrupt:
                print('Stop optimization\n')
                break

        try:
            all_loss = np.concatenate(all_loss)

        except:
            all_loss = np.array(all_loss, dtype=object)

        return all_loss


    def attack(self, train, test, meta, initialize_type='random', z_dim=128, lambda3=0.001, batch_size=2, maxfunc=3, lbfgs_lr=0.0001):

        wb = WhiteBox(self.synthesizer, meta, z_dim, lambda3=lambda3, batch_size=batch_size)

        ### initialization of z
        if initialize_type == 'zero':
            init_val = torch.zeros(len(train), z_dim)

            init_val_pos = init_val
            init_val_neg = init_val

        elif initialize_type == 'random':
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            np.random.seed(2021)

            mean_z = torch.zeros(len(train), z_dim)
            std_z = mean_z + 1
            init_val = torch.normal(mean=mean_z, std=std_z).to(device)

            init_val_pos = init_val
            init_val_neg = init_val
            
        else:
            raise NotImplementedError

        pos_query, neg_query = self.query(train, test)


        print('start pos at ', time.ctime(time.time()))
        pos_loss = self.optimize(wb, init_val_pos, pos_query, maxfunc, lbfgs_lr=lbfgs_lr)
        print("done pos at ", time.ctime(time.time()))
        print('start neg at ', time.ctime(time.time())) 
        neg_loss = self.optimize(wb, init_val_neg, neg_query, maxfunc, lbfgs_lr=lbfgs_lr)
        print('done neg at ', time.ctime(time.time()))

        labels = np.concatenate((np.zeros((len(neg_loss),)), np.ones((len(pos_loss),))))
        results = np.concatenate((-neg_loss, -pos_loss))

        auc = metrics.roc_auc_score(labels, results)
        ap = metrics.average_precision_score(labels, results)

        return auc, ap
