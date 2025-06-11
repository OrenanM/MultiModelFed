import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
import sys

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.datasets = args.datasets
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_sizes = args.batch_sizes
        self.learning_rate = args.local_learning_rate
        self.global_models = copy.deepcopy(args.models)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.ME = len(self.datasets)
        self.datasets_rate = np.array([1/self.ME for _ in range(self.ME)])

        self.rs_test_acc = [[] for i in range(self.ME)]
        self.rs_test_auc = [[] for i in range(self.ME)]
        self.rs_train_loss = [[] for i in range(self.ME)]

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new
        
        # Create Dataset
        self.balance = args.balance
        self.noniid = args.noniid
        self.partition_data = args.partition_data
        self.initial_alpha = args.initial_alpha

        # Concept Drift
        self.rounds_concept_drift = args.rounds_concept_drift[:]
        self.new_alpha = args.new_alpha[:]
        self.dataset_concept_drift = args.dataset_concept_drift

        # Label Shift
        self.rounds_label_shift = args.rounds_label_shift[:]
        self.dataset_label_shift = args.dataset_label_shift
        
        # Monitor acc
        self.acc_threthold = args.acc_threthold
        self.monitor_acc = args.monitor_acc

        # EMA
        self.alpha_ema = args.alpha_ema
        self.stability_global = [0 for _ in range(self.ME)]
        self.parameter_ema = args.parameter_ema # parameters, loss or acc
        if self.parameter_ema == 'parameters':
            self.stability_layers = []
            self.calculte_first_value_ema_parameters()
        self.rebalance_ema = args.rebalance_ema

        # first distribution
        self.concept_drift(alpha=self.initial_alpha)

    def calculte_first_value_ema_parameters(self):
        # for each model calculate the initial mean value for the EMA for each layer
 
        for me in range(self.ME): 
            stability = {} # dict mean parameters
            for name, paramaters in self.global_models[me].named_parameters():
                stability[name] = torch.mean(paramaters).item()
            
            # average of averages (average of the model) 
            mean_stability = torch.abs(torch.mean(torch.tensor(list(stability.values()))))
            
            # add values in list
            self.stability_layers.append(stability)
            self.stability_global[me] = mean_stability.item()

    def calculate_stability_parameters(self, me):

        ema_previous = self.stability_layers[me]

        for name, paramaters in self.global_models[me].named_parameters():
            ema = ema_previous[name]
            mean_paramaters = torch.mean(paramaters)

            new_ema = self.alpha_ema * mean_paramaters + (1-self.alpha_ema) * ema
            self.stability_layers[me][name] = new_ema.item()

        mean_stability = torch.abs(torch.mean(torch.tensor(list(self.stability_layers[me].values())))).item()
        print(self.stability_global[me])
        self.stability_global[me] = mean_stability
        print(self.stability_global[me])
    
    def calculate_stability_loss(self, me, round):
        if round == 0:
            self.stability_global[me] = self.rs_train_loss[me][-1]
            return
        ema = self.stability_global[me]
        new_ema = self.alpha_ema * self.rs_train_loss[me][-1] + (1-self.alpha_ema) * ema
        
        self.stability_global[me] = new_ema

    def calculate_stability_acc(self, me, round):
        if round == 0:
            self.stability_global[me] = self.rs_test_acc[me][-1]
            return
        ema = self.stability_global[me]
        new_ema = self.alpha_ema * self.rs_test_acc[me][-1] + (1-self.alpha_ema) * ema

        self.stability_global[me] = new_ema

    def models_rebalance_ema(self):
        for me in range(self.ME):
            self.datasets_rate[me] = self.stability_global[me]/sum(self.stability_global)

    def concept_drift(self, alpha=None):
        if alpha is None:
            print('=============== CONCEPT DRIFT ===============')
            alpha = self.new_alpha[0]
            # remove alpha da lista de alphas
            self.new_alpha.remove(self.new_alpha[0])

        # Cria distribução com novo alpha
        os.system(f'cd ../dataset && python generate_{self.dataset_concept_drift}.py \
                  {self.noniid} {self.balance} {self.partition_data} {alpha}')

    def shift_labels(self):
        # troca as labels de todos os clientes
        print('================ SHIFT LABEL ================ ')
        os.system(f'cd ../dataset && python generate_{self.dataset_concept_drift}.py \
                  {self.noniid} {self.balance} {self.partition_data} {self.initial_alpha}')

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = []
            test_data = []

            for me in range(self.ME):
                train = read_client_data(self.datasets[me], i, is_train=True, few_shot=self.few_shot)
                test = read_client_data(self.datasets[me], i, is_train=False, few_shot=self.few_shot)

                train_data.append(train)
                test_data.append(test)
            train_samples = []
            test_samples = []

            for train, test in zip(train_data, test_data):
                train_samples.append(len(train))
                test_samples.append(len(test_samples))

            client = clientObj(self.args, 
                        id=i, 
                        train_samples=train_samples, 
                        test_samples=test_samples, 
                        train_slow=train_slow, 
                        send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        print('*'*40)
        print(self.datasets_rate)
        print('*'*40)
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        selected = []
        for me in range(self.ME):
            # selection client for eat dataset
            num_me_select = int(self.current_num_join_clients * self.datasets_rate[me])
            selected_clients_me = list(np.random.choice(selected_clients, num_me_select, replace=False))
            selected.append(selected_clients_me)    

            selected_clients = [client for client in selected_clients if client not in selected_clients_me]
        return selected

    def send_models(self, me):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_models[me], me)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self, me):
        assert (len(self.selected_clients[me]) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.selected_clients[me]:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples[me]
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples[me])
                self.uploaded_models.append(client.models[me])
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self, me):
        assert (len(self.uploaded_models) > 0)

        self.global_models[me] = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_models[me].parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model, me)
        

    def add_parameters(self, w, client_model, me):
        for server_param, client_param in zip(self.global_models[me].parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self, me):
        model_path = os.path.join("models", self.datasets[me])
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_models[me], model_path)

    def load_model(self):
        model_path = os.path.join("models", self.datasets)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_models = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.datasets)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self, me):
        algo = self.datasets[me] + "_" + self.algorithm
        result_path = "../results/"

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc[me])):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc[me])
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc[me])
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss[me])

    def monitor_accuracy(self):
        for me in range(self.ME):
            if len(self.rs_test_acc[me]) >= 2:
                difference = self.rs_test_acc[me][-2] - self.rs_test_acc[me][-1] 
                print(self.rs_test_acc[me])
                print(difference)
                if difference > self.acc_threthold:
                    print('=============== SHIFT DETECTED ===============')
                    self.datasets_rate[me] *= 2

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self, me):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics(me)
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self, me):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics(me)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, me, acc=None, loss=None):
        stats = self.test_metrics(me)
        stats_train = self.train_metrics(me)

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc[me].append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss[me].append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_models.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.datasets, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.datasets, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_models)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
