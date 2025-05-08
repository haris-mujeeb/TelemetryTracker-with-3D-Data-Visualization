import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.linalg import inv

class OrientationKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state
        self.P = initial_covariance
        self.Q = process_noise
        self.R = measurement_noise
        self.F = np.eye(3)
        self.H = np.eye(3)

    def predict(self, gyro_rates, dt):
        self.state += gyro_rates * dt
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, accel_measurement, mag_measurement):
        z = np.array([accel_measurement[0], accel_measurement[1], mag_measurement])
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        self.state += K @ y
        I = np.eye(3)
        self.P = (I - K @ self.H) @ self.P
        return self.state

class KalmanOrientationEstimator:
    def __init__(
        self,
        accel_data,
        mag_data,
        gyro_data,
        timestamps_ms,
        alpha=0.95,
        initial_covariance=None,
        process_noise=None,
        measurement_noise=None
    ):
        self.accel_data = np.array(accel_data)
        self.mag_data = np.array(mag_data)
        self.gyro_data = np.array(gyro_data)
        self.timestamps_ms = np.array(timestamps_ms)
        self.dt_array = np.diff(self.timestamps_ms, prepend=self.timestamps_ms[0]) / 1000.0

        # Tuning constants
        self.alpha = alpha
        self.initial_cov = np.eye(3) * 0.01 if initial_covariance is None else initial_covariance
        self.process_noise = np.diag([0.01**2, 0.01**2, 0.02**2]) if process_noise is None else process_noise
        self.measurement_noise = np.diag([0.2**2, 0.2**2, 0.3**2]) if measurement_noise is None else measurement_noise

        # Pre-allocated arrays
        self.roll_from_accel = None
        self.pitch_from_accel = None
        self.yaw_from_mag = None

        self.roll_from_gyro = None
        self.pitch_from_gyro = None
        self.yaw_from_gyro = None

        self.roll_complementary = None
        self.pitch_complementary = None
        self.yaw_complementary = None

        self.roll_kalman = None
        self.pitch_kalman = None
        self.yaw_kalman = None

    def compute_angles(self):
        ax, ay, az = self.accel_data.T
        mx, my, mz = self.mag_data.T
        gx, gy, gz = self.gyro_data.T

        # Roll & Pitch from accelerometer
        self.roll_from_accel = np.arctan2(ay, az)
        self.pitch_from_accel = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

        # Normalize magnetometer
        mag_norm = np.linalg.norm(self.mag_data, axis=1)
        mx /= mag_norm
        my /= mag_norm
        mz /= mag_norm

        cos_r = np.cos(self.roll_from_accel)
        sin_r = np.sin(self.roll_from_accel)
        cos_p = np.cos(self.pitch_from_accel)
        sin_p = np.sin(self.pitch_from_accel)

        mx_comp = mx * cos_p + mz * sin_p
        my_comp = mx * sin_r * sin_p + my * cos_r - mz * sin_r * cos_p

        # Filter out invalid yaw readings
        mag_strength = np.sqrt(np.maximum(1e-8, mx_comp**2 + my_comp**2))
        yaw_temp = np.abs(np.arctan2(my_comp, mx_comp))
        self.yaw_from_mag = np.zeros(len(self.dt_array))

        valid_mag = mag_strength > 0.2
        yaw_var = uniform_filter1d(yaw_temp**2, size=10) - uniform_filter1d(yaw_temp, size=10)**2
        valid_var = yaw_var < 0.05
        valid = valid_mag & valid_var

        # Initialize yaw angle from magnetic data
        self.yaw_from_mag[0] = yaw_temp[0] if valid[0] else 0.0

        for i in range(1, len(self.yaw_from_mag)):
            # Use gyro z-axis (yaw rate) to predict the next yaw
            yaw_rate = self.gyro_data[i, 2]  # gZ is the yaw rate
            delta_time = self.dt_array[i]  # Time delta in seconds

            # Update yaw angle based on gyro data
            self.yaw_from_mag[i] = self.yaw_from_mag[i - 1] + yaw_rate * delta_time

            # Adjust yaw based on magnetic data validity
            if valid[i]:
                self.yaw_from_mag[i] = yaw_temp[i]  # Use magnetic yaw if valid
            else:
                self.yaw_from_mag[i] = self.yaw_from_mag[i - 1]  # Retain previous value if not valid



        # Complementary Filter
        n = len(self.dt_array)
        self.roll_from_gyro = np.zeros(n)
        self.pitch_from_gyro = np.zeros(n)
        self.yaw_from_gyro = np.zeros(n)

        self.roll_complementary = np.zeros(n)
        self.pitch_complementary = np.zeros(n)
        self.yaw_complementary = np.zeros(n)

        self.roll_complementary[0] = self.roll_from_accel[0]
        self.pitch_complementary[0] = self.pitch_from_accel[0]
        self.yaw_complementary[0] = self.yaw_from_mag[0]

        for i in range(1, n):
            dt = self.dt_array[i]
            self.roll_from_gyro[i] = self.roll_complementary[i - 1] + gx[i] * dt
            self.pitch_from_gyro[i] = self.pitch_complementary[i - 1] + gy[i] * dt
            self.yaw_from_gyro[i] = self.yaw_complementary[i - 1] + gz[i] * dt

            self.roll_complementary[i] = self.alpha * self.roll_from_gyro[i] + (1 - self.alpha) * self.roll_from_accel[i]
            self.pitch_complementary[i] = self.alpha * self.pitch_from_gyro[i] + (1 - self.alpha) * self.pitch_from_accel[i]
            self.yaw_complementary[i] = self.alpha * self.yaw_from_gyro[i] + (1 - self.alpha) * self.yaw_from_mag[i]

        # Kalman Filter
        self.roll_kalman = np.zeros(n)
        self.pitch_kalman = np.zeros(n)
        self.yaw_kalman = np.zeros(n)

        initial_state = np.array([self.roll_from_accel[0], self.pitch_from_accel[0], self.yaw_from_mag[0]])
        kalman_filter = OrientationKalmanFilter(
            initial_state=initial_state,
            initial_covariance=self.initial_cov,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise
        )

        self.roll_kalman[0], self.pitch_kalman[0], self.yaw_kalman[0] = initial_state

        for i in range(1, n):
            dt = self.dt_array[i]
            gyro_rates = np.array([gx[i], gy[i], gz[i]])
            kalman_filter.predict(gyro_rates, dt)
            kalman_filter.update([self.roll_from_accel[i], self.pitch_from_accel[i]], self.yaw_from_mag[i])
            self.roll_kalman[i], self.pitch_kalman[i], self.yaw_kalman[i] = kalman_filter.state

        return self.roll_kalman, self.pitch_kalman, self.yaw_kalman