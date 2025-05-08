import numpy as np
import pandas as pd

class SensorData:
    def __init__(self, data_dir, data_file):
        self.data_dir = data_dir
        self.data_file = data_file
        self.raw_data = None
        self.acc_raw_data = None
        self.gyro_raw_data = None
        self.mag_raw_data = None
        self.gps_raw_data = None
        self.timestamps_in_ms = None
        self.timestamps_in_seconds = None
        self.gps_timestamps_in_ms = None
        self.gps_timestamps_in_seconds = None
        self.time_deltas = None
        self.min_length = None

        self.load_data()

    def load_data(self):
        # Load the CSV file
        self.raw_data = pd.read_csv(self.data_dir + self.data_file)

        # Normalize timestamps
        self.raw_data["timestamp"] = self.raw_data["timestamp"] - self.raw_data["timestamp"][0]
        self.raw_data["IMU-ticks"] = self.raw_data["IMU-ticks"] - self.raw_data["IMU-ticks"][0]
        self.raw_data["GPS-ticks"] = self.raw_data["GPS-ticks"] - self.raw_data["GPS-ticks"][0]
        self.raw_data["pressure-ticks"] = self.raw_data["pressure-ticks"] - self.raw_data["pressure-ticks"][0]

        # Extract IMU data
        self.acc_raw_data = self.raw_data[['aX', 'aY', 'aZ']].to_numpy()  # m/s²
        self.gyro_raw_data = self.raw_data[['gX', 'gY', 'gZ']].to_numpy()  # rad/s
        self.mag_raw_data = self.raw_data[['mX', 'mY', 'mZ']].to_numpy()  # uT
        self.gps_raw_data = self.raw_data[['lat', 'long', 'alt', 'NedNorthVel', 'NedEastVel']].to_numpy() / 1e7
        
        self.mag_raw_data /= 1000  # Convert to mT
        self.timestamps_in_ms = self.raw_data['IMU-ticks'].to_numpy()
        self.timestamps_in_seconds = self.timestamps_in_ms / 1000.0
        self.time_deltas = np.diff(self.timestamps_in_ms) / 1000.0  # Convert ms to seconds
        self.time_deltas = np.append(self.time_deltas, self.time_deltas[-1])  # Same length as data
        self.gps_timestamps_in_ms = self.raw_data['GPS-ticks'].to_numpy()
        self.gps_timestamps_in_seconds = self.gps_timestamps_in_ms / 1000.0

        # Determine the minimum length among the arrays
        self.min_length = min(len(self.acc_raw_data), len(self.gyro_raw_data), 
                              len(self.mag_raw_data), len(self.gps_raw_data), 
                              len(self.timestamps_in_ms), len(self.time_deltas))

        # Trim all arrays to the minimum length
        self.acc_raw_data = self.acc_raw_data[:self.min_length]
        self.gyro_raw_data = self.gyro_raw_data[:self.min_length]
        self.mag_raw_data = self.mag_raw_data[:self.min_length]
        self.gps_raw_data = self.gps_raw_data[:self.min_length]
        self.timestamps_in_ms = self.timestamps_in_ms[:self.min_length]
        self.timestamps_in_seconds = self.timestamps_in_seconds[:self.min_length]
        self.time_deltas = self.time_deltas[:self.min_length]

        # Print average sampling period and frequency
        self.print_sampling_info()

    def print_sampling_info(self):
        avg_sampling_period = np.average(self.time_deltas)
        sampling_frequency = 1 / avg_sampling_period if avg_sampling_period > 0 else 0
        print(f"Average sampling period: {avg_sampling_period:.4f} seconds")
        print(f"Approximate sampling frequency: {sampling_frequency:.2f} Hz")

