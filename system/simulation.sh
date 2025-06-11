# Concept Drift
cd ../dataset
python generate_Cifar10.py noniid False dir 0.1
python generate_MNIST.py noniid False dir 0.1
cd ../system

## MNIST
python main.py -nc 100 -gr 30 --dataset_concept_drift='MNIST' --new_alpha=1 --initial_alpha=0.1 --rounds_concept_drift=15 -go concept_drift_mnist_0.1_1_rebalance_loss -pe loss -re
python main.py -nc 100 -gr 30 --dataset_concept_drift='MNIST' --new_alpha=1 --initial_alpha=0.1 --rounds_concept_drift=15 -go concept_drift_mnist_0.1_1_rebalance_acc -pe acc -re
python main.py -nc 100 -gr 30 --dataset_concept_drift='MNIST' --new_alpha=1 --initial_alpha=0.1 --rounds_concept_drift=15 -go concept_drift_mnist_0.1_1_rebalance_param -pe parameters -re

python main.py -nc 100 -gr 30 --dataset_concept_drift='MNIST' --new_alpha=0.1 --initial_alpha=1 --rounds_concept_drift=15 -go concept_drift_mnist_1_0.1_rebalance_loss -pe loss -re
python main.py -nc 100 -gr 30 --dataset_concept_drift='MNIST' --new_alpha=0.1 --initial_alpha=1 --rounds_concept_drift=15 -go concept_drift_mnist_1_0.1_rebalance_acc -pe acc -re
python main.py -nc 100 -gr 30 --dataset_concept_drift='MNIST' --new_alpha=0.1 --initial_alpha=1 --rounds_concept_drift=15 -go concept_drift_mnist_1_0.1_rebalance_param -pe parameters -re


## Cifar10
python main.py -nc 100 -gr 100 --dataset_concept_drift='Cifar10' --new_alpha=1 --initial_alpha=0.1 --rounds_concept_drift=50 -go concept_drift_cifar10_0.1_1_loss -pe loss -re
python main.py -nc 100 -gr 100 --dataset_concept_drift='Cifar10' --new_alpha=1 --initial_alpha=0.1 --rounds_concept_drift=50 -go concept_drift_cifar10_0.1_1_acc -pe acc -re
python main.py -nc 100 -gr 100 --dataset_concept_drift='Cifar10' --new_alpha=1 --initial_alpha=0.1 --rounds_concept_drift=50 -go concept_drift_cifar10_0.1_1_param -pe parameters -re

python main.py -nc 100 -gr 100 --dataset_concept_drift='Cifar10' --new_alpha=0.1 --initial_alpha=1 --rounds_concept_drift=50 -go concept_drift_cifar10_1_0.1_loss -pe loss -re
python main.py -nc 100 -gr 100 --dataset_concept_drift='Cifar10' --new_alpha=0.1 --initial_alpha=1 --rounds_concept_drift=50 -go concept_drift_cifar10_1_0.1_acc -pe acc -re
python main.py -nc 100 -gr 100 --dataset_concept_drift='Cifar10' --new_alpha=0.1 --initial_alpha=1 --rounds_concept_drift=50 -go concept_drift_cifar10_1_0.1_param -pe parameters -re

######################################################################

# Label Shift
cd ../dataset
python generate_Cifar10.py noniid False dir 0.1
python generate_MNIST.py noniid False dir 0.1
cd ../system

## MNIST
python main.py -nc 100 -gr 30 --dataset_label_shift='MNIST' --rounds_label_shift=15 -go data_shift_mnist_rebalance_acc   -pe acc        -re
python main.py -nc 100 -gr 30 --dataset_label_shift='MNIST' --rounds_label_shift=15 -go data_shift_mnist_rebalance_loss  -pe loss       -re
python main.py -nc 100 -gr 30 --dataset_label_shift='MNIST' --rounds_label_shift=15 -go data_shift_mnist_rebalance_param -pe parameters -re
## Cifar10
python main.py -nc 100 -gr 100 --dataset_label_shift='Cifar10' --rounds_label_shift=50 -go data_shift_cifar10_rebalance_acc   -pe acc        -re
python main.py -nc 100 -gr 100 --dataset_label_shift='Cifar10' --rounds_label_shift=50 -go data_shift_cifar10_rebalance_loss  -pe loss       -re
python main.py -nc 100 -gr 100 --dataset_label_shift='Cifar10' --rounds_label_shift=50 -go data_shift_cifar10_rebalance_param -pe parameters -re