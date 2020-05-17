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
EPOCHS = 40
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
	general_model = tf.keras.models.Sequential()
	general_model.add(tf.keras.layers.LSTM(128, input_shape=(HISTORY, 12), return_sequences=True, activation='tanh'))
	general_model.add(tf.keras.layers.Dropout(0.2))
	general_model.add(tf.keras.layers.LSTM(128, input_shape=(HISTORY, 12), return_sequences=True, activation='tanh'))
	general_model.add(tf.keras.layers.Dropout(0.2))
	general_model.add(tf.keras.layers.LSTM(128, input_shape=(HISTORY, 12), activation='tanh'))
	general_model.add(tf.keras.layers.Dense(32, activation='relu'))
	general_model.add(tf.keras.layers.Dense(1))

	general_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

	return general_model

def create_paths(paths):
	for path in paths:
		if not os.path.exists(path):
			try:
				os.makedirs(path)
			except OSError:
				print(f"{path} already exists")
			else:
				print(f"Successfully created the directory {path}")

def train_machine_general(df_machine, general_model, target_index):
	print(f"Training for {df_machine[list(df_machine.keys())[0]].columns[target_index]} based on:\n{df_machine[list(df_machine.keys())[0]].columns}")
	target_index_name = df_machine[list(df_machine.keys())[0]].columns[target_index]

	# Create all required filepaths
	main_path = f"models"
	model_path = f"{main_path}/general_h5"
	model_machine_path = f"{model_path}/{target_index_name}"
	figures_path = f"{model_machine_path}/figures"
	create_paths([main_path, model_path, model_machine_path, figures_path])

	x_train_single, y_train_single = np.array([]), np.array([])
	x_val_single, y_val_single = np.array([]), np.array([])


	allData = np.array([])
	print(f"-----GENERATING ALL DATA-----")
	for machine in tqdm(df_machine.keys()):
		TRAIN_SPLIT_INT = int(len(df_machine[machine].values) * TRAIN_SPLIT)

		if allData.size == 0:
			allData = df_machine[machine].values[:TRAIN_SPLIT_INT]
		else:
			allData = np.concatenate((allData, df_machine[machine].values[:TRAIN_SPLIT_INT]))

	data_mean = allData.mean(axis=0)
	data_std = allData.std(axis=0)

	# Save mean and std for deployment
	np.savetxt(f"{model_machine_path}/data_mean.csv", data_mean, delimiter=",")
	np.savetxt(f"{model_machine_path}/data_std.csv", data_std, delimiter=",")

	print(f"-----CREATING DATA GENERATORS-----")
	for machine in tqdm(df_machine.keys()):
		# Load the dataset as an array
		dataset = df_machine[machine].values

		# Split training and testing data
		TRAIN_SPLIT_INT = int(len(dataset) * TRAIN_SPLIT)

		# Normalize the data
		dataset = (dataset - data_mean) / data_std

		# Split the data into training and testing and X and y
		x_train_single_tmp, y_train_single_tmp = multivariate_data(dataset, dataset[:, target_index],
															0, TRAIN_SPLIT_INT,
															HISTORY, 1, 1,
                                                   			single_step=True)
		x_val_single_tmp, y_val_single_tmp = multivariate_data(dataset, dataset[:, target_index],
                                               				TRAIN_SPLIT_INT, None,
                                               				HISTORY, 1, 1,
                                               				single_step=True)

		if x_train_single.size == 0:
			x_train_single = x_train_single_tmp
		else:
			x_train_single = np.concatenate((x_train_single, x_train_single_tmp))
		if y_train_single.size == 0:
			y_train_single = y_train_single_tmp
		else:
			y_train_single = np.concatenate((y_train_single, y_train_single_tmp))
		if x_val_single.size == 0:
			x_val_single = x_val_single_tmp
		else:
			x_val_single = np.concatenate((x_val_single, x_val_single_tmp))
		if y_val_single.size == 0:
			y_val_single = y_val_single_tmp
		else:
			y_val_single = np.concatenate((y_val_single, y_val_single_tmp))

	train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
	train_data_single = train_data_single.cache().shuffle(len(allData) * 2).batch(BATCH_SIZE).repeat()

	val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
	val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

	log_dir = f"general_h5_logs/fit/general_{target_index_name}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
	checkpoint_val_loss = tf.keras.callbacks.ModelCheckpoint(f"{model_machine_path}/model_best_val_loss.h5", monitor="val_loss", save_best_only=True, mode="min")
	checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(f"{model_machine_path}/model_best_loss.h5", monitor="loss", save_best_only=True, mode="min")

	general_history = general_model.fit(train_data_single, epochs=EPOCHS,
											steps_per_epoch=len(allData),
											validation_data=val_data_single,
											validation_steps=50,
											callbacks=[tensorboard, checkpoint_val_loss, checkpoint_loss])

	# convert the history.history dict to a pandas DataFrame:     
	hist_df = pd.DataFrame(general_history.history) 

	# or save to csv: 
	hist_csv_file = f"{model_machine_path}/history.csv"
	with open(hist_csv_file, mode='w') as f:
		hist_df.to_csv(f)

	general_model.save(f"{model_machine_path}/model.h5")
	
	best_model = tf.keras.models.load_model(f"{model_machine_path}/model_best_val_loss.h5")

	val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
	val_data_single = val_data_single.shuffle(len(allData) * 2).batch(BATCH_SIZE).repeat()
	index = 0
	plt.clf()
	for x, y in val_data_single.take(20):
		plot = show_plot([x[0][:, target_index].numpy(), y[0].numpy(), best_model.predict(x)[0]], 1, 'General model prediction (validation data)')
		plot.savefig(f"{model_machine_path}/figures/validation_{index}.pdf")
		plt.clf()
		index += 1

	index = 0
	plt.clf()
	for x, y in train_data_single.take(100):
		plot = show_plot([x[0][:, target_index].numpy(), y[0].numpy(), best_model.predict(x)[0]], 1, 'General model prediction (training data)')
		plot.savefig(f"{model_machine_path}/figures/training_{index}.pdf")
		plot.clf()
		index += 1

df_machine = load_data()

# Remove data from the period when the sensors were calibrated
df_machine["FL01_before"] = df_machine["FL01"][df_machine["FL01"].index <= "2019-03-10"]
df_machine["FL01_after"] = df_machine["FL01"][df_machine["FL01"].index >= "2019-03-20"]
del df_machine["FL01"]

df_machine["FL07_before"] = df_machine["FL07"][df_machine["FL07"].index <= "2019-03-10"]
df_machine["FL07_after"] = df_machine["FL07"][df_machine["FL07"].index >= "2019-03-20"]
del df_machine["FL07"]

for i in range(0, 12, 1):
	general_model = get_model()
	train_machine_general(df_machine, general_model, i)




