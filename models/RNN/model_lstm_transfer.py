import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import datetime
import os
import json
'''
HYPERPARAMETERS
'''
TIME_STEPS = 7
EPOCHS = 60
BATCH_SIZE = 1
TRAIN_SPLIT = 0.7
HISTORY = 30

plt.style.use('fast')

def load_data():
	print("-----LOADING THE DATASET-----")
	df = pd.read_csv('../../dataset.csv', sep=';')
	df["date_measurement"] = pd.to_datetime(df["date_measurement"])
	df["start_timestamp"] = pd.to_datetime(df["start_timestamp"])
	df["end_timestamp"] = pd.to_datetime(df["end_timestamp"])
	df["duration"] = df["end_timestamp"] - df["start_timestamp"]

	machines = df["machine_name"].unique()
	print("Found the following machines:", machines)
	df_machine = {}
	print("-----REARRANGING THE DATA-----")
	for machine in tqdm(machines):
		df_machine[machine] = df[df["machine_name"] == machine].copy()
		df_machine[machine].set_index("date_measurement", inplace=True)
		df_machine[machine] = df_machine[machine].pivot_table(index="date_measurement", columns="sensor_type", values="realvalue")
		df_machine[machine].sort_index()

		df_machine[machine] = df_machine[machine].reindex(pd.date_range(df_machine[machine].index.min(), df_machine[machine].index.max()))
		for col in df_machine[machine].columns:
			df_machine[machine][col] = df_machine[machine][col].interpolate()
			df_machine[machine][col] = df_machine[machine][col].rolling(7, min_periods=1).mean()

	return df_machine

def graph_sensors(df):
	print(f"-----GENERATING GRAPHS-----")
	for sensor in tqdm(df.columns):
		df.reset_index().plot("date_measurement", f"{sensor}", title=sensor)
	plt.show()

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
	data = []
	labels = []

	start_index = start_index + history_size
	if end_index is None:
		end_index = len(dataset) - target_size

	for i in range(start_index, end_index):
		indices = range(i-history_size, i, step)
		data.append(dataset[indices])

		if single_step:
			labels.append(target[i+target_size])
		else:
			labels.append(target[i:i+target_size])

	return np.array(data), np.array(labels)

def plot_train_history(history, title, machine):
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(len(loss))

	plt.figure()

	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title(title)
	plt.legend()

	#plt.show()
	plt.savefig(f"models/single/{machine}_deep_advanced/figures/train_history.pdf")

def plot_train_history_2(history, title, machine):
	loss = history.history['loss']

	epochs = range(len(loss))

	plt.figure()

	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.title(title)
	plt.legend()

	#plt.show()
	plt.savefig(f"models/single/{machine}_deep_advanced/figures/train_history.pdf")

def create_time_steps(length):
	return list(range(-length, 0))

def show_plot(plot_data, delta, title):
	labels = ['History', 'True Future', 'Model Prediction']
	marker = ['.-', 'rx', 'go']
	time_steps = create_time_steps(plot_data[0].shape[0])
	if delta:
		future = delta
	else:
		future = 0

	plt.title(title)
	for i, x in enumerate(plot_data):
		if i > 0:
			plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
		elif i == 0:
			plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
	plt.legend()
	plt.xlim([time_steps[0], (future+5)*2])
	plt.xlabel('Time-Step')
	return plt

def get_model():
	single_step_model = tf.keras.models.Sequential()
	single_step_model.add(tf.keras.layers.LSTM(128, input_shape=(HISTORY, 12), return_sequences=True, activation='tanh'))
	single_step_model.add(tf.keras.layers.Dropout(0.2))
	single_step_model.add(tf.keras.layers.LSTM(128, input_shape=(HISTORY, 12), return_sequences=True, activation='tanh'))
	single_step_model.add(tf.keras.layers.Dropout(0.2))
	single_step_model.add(tf.keras.layers.LSTM(128, input_shape=(HISTORY, 12), activation='tanh'))
	single_step_model.add(tf.keras.layers.Dense(32, activation='relu'))
	single_step_model.add(tf.keras.layers.Dense(1))

	single_step_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

	return single_step_model

def create_paths(paths):
	for path in paths:
		if not os.path.exists(path):
			try:
				os.makedirs(path)
			except OSError:
				print(f"{path} already exists")
			else:
				print(f"Successfully created the directory {path}")

def train_machine(df_machine, machine, transfer_model, target_index):
	print(f"Training for {df_machine[machine].columns[target_index]} based on:\n{df_machine[machine].columns}")
	target_index_name = df_machine[machine].columns[target_index]

	# Create all required filepaths
	main_path = f"models"
	model_path = f"{main_path}/transfer"
	model_1_path = f"{model_path}/{machine}"
	model_machine_path = f"{model_1_path}/{target_index_name}"
	figures_path = f"{model_machine_path}/figures"
	create_paths([main_path, model_path, model_1_path, model_machine_path, figures_path])

	# Load the dataset as an array
	dataset = df_machine[machine].values

	# Split training and testing data
	TRAIN_SPLIT_INT = int(len(dataset) * TRAIN_SPLIT)

	# Load mean and std
	data_mean = np.loadtxt(open(f"models/general_h5/{target_index_name}/data_mean.csv", "rb"), delimiter=",")
	data_std = np.loadtxt(open(f"models/general_h5/{target_index_name}/data_std.csv", "rb"), delimiter=",")

	# Normalize the data
	dataset = (dataset - data_mean) / data_std

	# Split the data into training and testing and X and y
	x_train_single, y_train_single = multivariate_data(dataset, dataset[:, target_index],
														0, TRAIN_SPLIT_INT,
														HISTORY, 1, 1,
                                                   single_step=True)
	x_val_single, y_val_single = multivariate_data(dataset, dataset[:, target_index],
                                               TRAIN_SPLIT_INT, None,
                                               HISTORY, 1, 1,
                                               single_step=True)

	train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
	train_data_single = train_data_single.cache().shuffle(len(dataset) * 2).batch(BATCH_SIZE).repeat()

	val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
	val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

	log_dir = f"transfer_logs/fit/{machine}_{target_index_name}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
	checkpoint_val_loss = tf.keras.callbacks.ModelCheckpoint(f"{model_machine_path}/model_best_val_loss.h5", monitor="val_loss", save_best_only=True, mode="min")
	checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(f"{model_machine_path}/model_best_loss.h5", monitor="loss", save_best_only=True, mode="min")
	
	transfer_history = transfer_model.fit(train_data_single, epochs=EPOCHS,
											steps_per_epoch=len(dataset),
											validation_data=val_data_single,
											validation_steps=50,
											callbacks=[tensorboard, checkpoint_loss, checkpoint_val_loss])
	transfer_model.save(f"{model_machine_path}/{machine}.h5")

	# convert the history.history dict to a pandas DataFrame:     
	hist_df = pd.DataFrame(transfer_history.history) 

	# or save to csv: 
	hist_csv_file = f"{model_machine_path}/history.csv"
	with open(hist_csv_file, mode='w') as f:
		hist_df.to_csv(f)
	
	best_model = tf.keras.models.load_model(f"{model_machine_path}/model_best_val_loss.h5")

	val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
	val_data_single = val_data_single.shuffle(len(dataset) * 2).batch(BATCH_SIZE).repeat()
	index = 0
	plt.clf()
	for x, y in val_data_single.take(20):
		plot = show_plot([x[0][:, target_index].numpy(), y[0].numpy(), best_model.predict(x)[0]], 1, 'Single Step Prediction (validation data)')
		plot.savefig(f"{model_machine_path}/figures/validation_{index}.pdf")
		plt.clf()
		index += 1

	index = 0
	plt.clf()
	for x, y in train_data_single.take(100):
		plot = show_plot([x[0][:, target_index].numpy(), y[0].numpy(), best_model.predict(x)[0]], 1, 'Single Step Prediction (training data)')
		plot.savefig(f"{model_machine_path}/figures/training_{index}.pdf")
		plot.clf()
		index += 1

df_machine = load_data()
#df_machine["FL06"].plot(subplots=True)
plt.show()

# Remove data from the period when the sensors were calibrated
df_machine["FL01_before"] = df_machine["FL01"][df_machine["FL01"].index <= "2019-03-10"]
df_machine["FL01_after"] = df_machine["FL01"][df_machine["FL01"].index >= "2019-03-20"]
del df_machine["FL01"]

df_machine["FL07_before"] = df_machine["FL07"][df_machine["FL07"].index <= "2019-03-10"]
df_machine["FL07_after"] = df_machine["FL07"][df_machine["FL07"].index >= "2019-03-20"]
del df_machine["FL07"]

print(df_machine["FL02"].columns)

for i in range(0, 12, 1):
	target_index_name = df_machine[list(df_machine.keys())[0]].columns[i]
	main_path = f"models"
	model_path = f"{main_path}/general_h5"
	model_machine_path = f"{model_path}/{target_index_name}"

	transfer_model = tf.keras.models.load_model(f"{model_machine_path}/model.h5")
	train_machine(df_machine, f"FL01_before", transfer_model, i)
	train_machine(df_machine, f"FL01_after", transfer_model, i)

for m in range(2, 7):
	for i in range(0, 12, 1):
		target_index_name = df_machine[list(df_machine.keys())[0]].columns[i]
		main_path = f"models"
		model_path = f"{main_path}/general_h5"
		model_machine_path = f"{model_path}/{target_index_name}"

		transfer_model = tf.keras.models.load_model(f"{model_machine_path}/model.h5")
		print(transfer_model.summary())
		train_machine(df_machine, f"FL0{m}", transfer_model, i)

for i in range(0, 12, 1):
	target_index_name = df_machine[list(df_machine.keys())[0]].columns[i]
	main_path = f"models"
	model_path = f"{main_path}/general_h5"
	model_machine_path = f"{model_path}/{target_index_name}"

	transfer_model = tf.keras.models.load_model(f"{model_machine_path}/model.h5")
	train_machine(df_machine, f"FL07_before", transfer_model, i)
	train_machine(df_machine, f"FL07_after", transfer_model, i)