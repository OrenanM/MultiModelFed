# Concept Drift
cd ../dataset
python generate_Cifar10.py noniid False dir 0.1
python generate_MNIST.py noniid False dir 0.1
cd ../system

## Cifar10
python main.py -nc 100 -gr 100 --dataset_concept_drift='Cifar10' -t 5 --new_alpha=0.1 --initial_alpha=1 --rounds_concept_drift=50 -go concept_drift_cifar10_1_0.1
python main.py -nc 100 -gr 100 --dataset_concept_drift='Cifar10' -t 5 --new_alpha=0.1 --initial_alpha=1 --rounds_concept_drift=50 -go concept_drift_cifar10_1_0.1_rebalance -ma
python main.py -nc 100 -gr 100 --dataset_concept_drift='Cifar10' -t 5 --new_alpha=1 --initial_alpha=0.1 --rounds_concept_drift=50 -go concept_drift_cifar10_0.1_1
python main.py -nc 100 -gr 100 --dataset_concept_drift='Cifar10' -t 5 --new_alpha=1 --initial_alpha=0.1 --rounds_concept_drift=50 -go concept_drift_cifar10_0.1_1_rebalance -ma

######################################################################

# Label Shift
cd ../dataset
python generate_Cifar10.py noniid False dir 0.1
python generate_MNIST.py noniid False dir 0.1
cd ../system

## MNIST
python main.py -nc 100 -gr 30 --dataset_label_shift='MNIST' -t 5 --rounds_label_shift=15 -go data_shift_mnist
python main.py -nc 100 -gr 30 --dataset_label_shift='MNIST' -t 5 --rounds_label_shift=15 -go data_shift_mnist_rebalance -ma
## Cifar10
python main.py -nc 100 -gr 100 --dataset_label_shift='Cifar10' -t 5 --rounds_label_shift=50 -go data_shift_cifar10
python main.py -nc 100 -gr 100 --dataset_label_shift='Cifar10' -t 5 --rounds_label_shift=50 -go data_shift_cifar10_rebalance -ma