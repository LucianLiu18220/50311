

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os


def showfig(vul_ann, result, tainingtime):
	print("training time: {0:.2f} minutes".format(tainingtime / 60))
	num_epochs = len(result)

	train_loss = result['train_loss']
	valid_loss = result['valid_loss']
	test_loss = result['test_loss']

	train_Accu = result['train_Accu']
	valid_Accu = result['valid_Accu']
	test_Accu = result['test_Accu']

	train_Prec = result['train_Prec']
	valid_Prec = result['valid_Prec']
	test_Prec = result['test_Prec']

	train_Reca = result['train_Reca']
	valid_Reca = result['valid_Reca']
	test_Reca = result['test_Reca']

	train_Spec = result['train_Spec']
	valid_Spec = result['valid_Spec']
	test_Spec = result['test_Spec']

	train_F1Sc = result['train_F1Sc']
	valid_F1Sc = result['valid_F1Sc']
	test_F1Sc = result['test_F1Sc']

	# print figure
	plt.figure(num=1, figsize=(15, 10))

	x = range(0, num_epochs)
	plt.subplot(3, 1, 1)
	plt.plot(x, train_Accu, color='red', linewidth=1, linestyle='solid', label='Trainset accuracy')
	plt.plot(x, valid_Accu, color='red', linewidth=1, linestyle='dotted', label='Validset accuracy')
	plt.plot(x, test_Accu, color='red', linewidth=1, linestyle='dashed', label='Testset accuracy')
	plt.plot(x, train_Prec, color='slategrey', linewidth=1, linestyle='solid', label='Trainset precision')
	plt.plot(x, valid_Prec, color='slategrey', linewidth=1, linestyle='dotted', label='Validset precision')
	plt.plot(x, test_Prec, color='slategrey', linewidth=1, linestyle='dashed', label='Testset precision')
	plt.plot(x, train_F1Sc, color='blue', linewidth=1, linestyle='solid', label='Trainset F1-score')
	plt.plot(x, valid_F1Sc, color='blue', linewidth=1, linestyle='dotted', label='Validset F1-score')
	plt.plot(x, test_F1Sc, color='blue', linewidth=1, linestyle='dashed', label='Testset F1-score')
	plt.grid(True)
	plt.legend()
	plt.title(f'metric vs. epoch ({vul_ann})')
	# plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.ylim((0, 1))

	plt.subplot(3, 1, 2)
	plt.plot(x, train_Reca, color='fuchsia', linewidth=1, linestyle='solid', label='Trainset recall')
	plt.plot(x, valid_Reca, color='fuchsia', linewidth=1, linestyle='dotted', label='Validset recall')
	plt.plot(x, test_Reca, color='fuchsia', linewidth=1, linestyle='dashed', label='Testset recall')
	plt.plot(x, train_Spec, color='green', linewidth=1, linestyle='solid', label='Trainset specificity')
	plt.plot(x, valid_Spec, color='green', linewidth=1, linestyle='dotted', label='Validset specificity')
	plt.plot(x, test_Spec, color='green', linewidth=1, linestyle='dashed', label='Testset specificity')
	plt.grid(True)
	plt.legend()
	plt.title(f'metric vs. epoch ({vul_ann})')
	# plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.ylim((0, 1))

	plt.subplot(3, 1, 3)
	plt.plot(x, train_loss, color='black', linewidth=1, linestyle='solid', label='Trainset Loss')
	plt.plot(x, valid_loss, color='black', linewidth=1, linestyle='dotted', label='Validset Loss')
	plt.plot(x, test_loss, color='black', linewidth=1, linestyle='dashed', label='Testset Loss')
	plt.plot(x, valid_loss - train_loss, color='black', linewidth=1, linestyle='dashdot', label='Validset Loss - Trainset Loss')
	plt.grid(True)
	plt.legend()
	plt.title(f'loss vs. epoch ({vul_ann})')
	plt.ylabel('average loss per sample')
	plt.xlabel('epoch')
	plt.ylim((0, 1))

	plt.tight_layout()
	plt.show()


vul_ann = "Tim"
method = "SCFG_DIHGNN"

Log_path = "./Log/" + vul_ann
Log_path_Current = Log_path + "/Log_" + str(method)

file_name = Log_path_Current + "/conf.log"
with open(file_name, 'r', encoding="UTF-8") as log_file:
	for line in log_file:
		print(line.strip())

path = Log_path_Current + "/trainingresult.pkl"
[tainingtime, result, alpha_list] = pickle.load(open(path, 'rb'))

print("**************************************************************************************")
current_file_path = os.path.abspath(__file__)
current_file_path = os.path.dirname(current_file_path)
current_file_name = os.path.basename(current_file_path)
print(f"model name: {current_file_name}")
print(f"{vul_ann}/log_{method}")

# Use the loss of the validation set as the early stopping criterion
valid_loss = [float(valid_loss) for valid_loss in result['valid_loss']]
min_valid_loss = min(valid_loss)
min_valid_loss_Epoch = max(i for i, v in enumerate(valid_loss) if v == min_valid_loss)
print("min valid loss Epoch: ", min_valid_loss_Epoch)

Epoch_data = result.iloc[min_valid_loss_Epoch]
print(f'【Epoch: {min_valid_loss_Epoch:02d}】')
print(f"\talpha: {alpha_list['alpha'][min_valid_loss_Epoch]:.4f}")
print(f"\talpha_list: {[alpha for alpha in list(alpha_list['alpha'])[:min_valid_loss_Epoch+1]]}")
print("\t{:^19}   |   {:^18}   |   {:^18}".format("Train", "Valid", "Test"))
print(f"\tTP: {Epoch_data['train_TP']:=5d}, FP: {Epoch_data['train_FP']:=5d}  |  TP: {Epoch_data['valid_TP']:=5d}, FP: {Epoch_data['valid_FP']:=5d}  |  TP: {Epoch_data['test_TP']:=5d}, FP: {Epoch_data['test_FP']:=5d}")
print(f"\tFN: {Epoch_data['train_FN']:=5d}, TN: {Epoch_data['train_TN']:=5d}  |  FN: {Epoch_data['valid_FN']:=5d}, TN: {Epoch_data['valid_TN']:=5d}  |  FN: {Epoch_data['test_FN']:=5d}, TN: {Epoch_data['test_TN']:=5d}")
print(f"\tTrain -- Accu: {Epoch_data['train_Accu']:.4f}, Prec: {Epoch_data['train_Prec']:.4f}, Reca: {Epoch_data['train_Reca']:.4f}, Spec: {Epoch_data['train_Spec']:.4f}, F1Sc: {Epoch_data['train_F1Sc']:.4f}")
print(f"\tValid -- Accu: {Epoch_data['valid_Accu']:.4f}, Prec: {Epoch_data['valid_Prec']:.4f}, Reca: {Epoch_data['valid_Reca']:.4f}, Spec: {Epoch_data['valid_Spec']:.4f}, F1Sc: {Epoch_data['valid_F1Sc']:.4f}")
print(f"\tTest  -- Accu: {Epoch_data['test_Accu']:.4f}, Prec: {Epoch_data['test_Prec']:.4f}, Reca: {Epoch_data['test_Reca']:.4f}, Spec: {Epoch_data['test_Spec']:.4f}, F1Sc: {Epoch_data['test_F1Sc']:.4f}")
print(f"\tAverage train Loss: {Epoch_data['train_loss']:.7f}")
print(f"\tAverage valid Loss: {Epoch_data['valid_loss']:.7f}")
print(f"\tAverage test  Loss: {Epoch_data['test_loss']:.7f}")

print("**************************************************************************************")


showfig(vul_ann, result, tainingtime)

print("a")


