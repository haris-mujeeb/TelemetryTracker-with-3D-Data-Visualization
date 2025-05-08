from sensor_data_loadder import SensorData
from orientation_estimation import KalmanOrientationEstimator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymap3d import geodetic2enu
# from pyproj import Proj, Transformer

DATA_DIR = './walks/outside_elab/new/'
DATA_FILE = 'trax_2025_4_25_3_37_16_#28.csv'

sensorData = SensorData(data_dir=DATA_DIR, data_file=DATA_FILE)

initial_covariance = np.eye(3) * 0.1**2  # Example values for position/angle uncertainty
process_noise = np.diag([0.01**2, 0.01**2, 0.02**2])  # Accelerometer, Gyroscope, Magnetometer
measurement_noise = np.diag([0.2**2, 0.2**2, 0.3**2])  # Accelerometer, Gyroscope, Magnetometer
alpha = 0.7

kalmanFilter = KalmanOrientationEstimator(
    sensorData.acc_raw_data,
    sensorData.mag_raw_data,
    sensorData.gyro_raw_data,
    sensorData.timestamps_in_ms,
    initial_covariance=initial_covariance,
    process_noise=process_noise,
    measurement_noise=measurement_noise,
    alpha=alpha
)
roll_kf, pitch_kf, yaw_kf = kalmanFilter.compute_angles()


# Example: GPS raw data: (N, 3) where each row is [lat, lon, alt]
# gps_raw_data = np.array([[lat1, lon1, alt1], [lat2, lon2, alt2], ..., [latN, lonN, altN]])
lat0, lon0, alt0, NedNorthVel0, NedEastVel0  = sensorData.gps_raw_data[0]  # Origin for ENU conversion

# Initialize arrays to store ENU coordinates
e = np.zeros(sensorData.gps_raw_data.shape[0])  # East
n = np.zeros(sensorData.gps_raw_data.shape[0])  # North
u = np.zeros(sensorData.gps_raw_data.shape[0])  # Upv

# Convert each GPS point (lat, lon, alt) to ENU
for i, (lat, lon, alt, nedNorthVel, nedEastVel) in enumerate(sensorData.gps_raw_data):
    e[i], n[i], u[i] = geodetic2enu(lat, lon, alt, lat0, lon0, alt0)

gps_unique_timestamps_in_seconds = np.unique(sensorData.gps_timestamps_in_seconds)

# Stack the results into a single (N, 3) array for ENU positions
# Interpolation
interpolated_e = np.interp(sensorData.timestamps_in_seconds, sensorData.gps_timestamps_in_seconds, e)
interpolated_n = np.interp(sensorData.timestamps_in_seconds, sensorData.gps_timestamps_in_seconds, n)

# Angles are defined as follows
angles = np.array([roll_kf, pitch_kf, yaw_kf])  # roll, pitch, yaw
angles = np.transpose(angles)

# Assume yaw_kf is in radians
yaw_unit_x = np.cos(yaw_kf)
yaw_unit_y = np.sin(yaw_kf)

# Every 20th index
idx = np.arange(0, len(interpolated_e), 20)

# Plotting
plt.figure(figsize=(12, 6))

# Plot e vs n
plt.subplot(1, 2, 1)
plt.plot(interpolated_e, interpolated_n, '-', label='Interpolated', linewidth=2)
plt.plot(e, n, 'o', label='Original e', markersize=1)

# Plot unit yaw vectors as quivers
plt.quiver(interpolated_e[idx], interpolated_n[idx], 
           yaw_unit_x[idx], yaw_unit_y[idx], 
           angles='xy', scale_units='xy', color='r', label='Yaw Unit Vectors', scale=0.5)

plt.title('e vs n')
plt.xlabel('n (North)')
plt.ylabel('e (East)')
plt.legend()
plt.grid()
plt.show()