import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import sys
import os

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):

            if i in self.rounds_concept_drift:
                self.create_new_distribution()
            if i in self.rounds_label_shift:
                self.shift_labels()

            s_t = time.time()
            self.selected_clients = self.select_clients()
            print(f"\n-------------Round number: {i}-------------")
            for me in range(self.ME):
                self.send_models(me)
                
                if i%self.eval_gap == 0:
                    print(f"\nEvaluate global model - {self.datasets[me]}")
                    self.evaluate(me)

                for client in self.selected_clients[me]:
                    client.train(me)

                self.receive_models(me)
                if self.dlg_eval and i%self.dlg_gap == 0:
                    self.call_dlg(i)
                self.aggregate_parameters(me)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        for me, dataset in enumerate(self.datasets):
            print(f'{dataset}: {max(self.rs_test_acc[me])}')
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        for me in range(self.ME):
            self.save_results(me)
            self.save_global_model(me)

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
