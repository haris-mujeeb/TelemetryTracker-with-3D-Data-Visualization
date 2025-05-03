import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ahrs
from ahrs.common.orientation import q2R
import os

def list_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

# Example usage
directory_path = './'
csv_files = list_csv_files(directory_path)

for file in csv_files:
    print(file)
    # Load data


    df = pd.read_csv(file)
    df["timestamp"] = df["timestamp"]- df["timestamp"][0]
    df["IMU-ticks"] = df["IMU-ticks"]- df["IMU-ticks"][0]
    df["GPS-ticks"] = df["GPS-ticks"]- df["GPS-ticks"][0]
    df["pressure-ticks"] = df["pressure-ticks"]- df["pressure-ticks"][0]

    df['lat'] = df['lat'] / 1e7
    df['long'] = df['long'] / 1e7


    print(df.head()) # to display the first 5 lines of loaded data


    # Extract IMU data
    acc = df[['aX', 'aY', 'aZ']].to_numpy() #m/s2
    gyr = df[['gX', 'gY', 'gZ']].to_numpy() #rad/s
    mag = df[['mX', 'mY', 'mZ']].to_numpy() #uT

    mag /= 1000 # the library expects mT
    ts = df['IMU-ticks'].to_numpy()
    dt = np.diff(ts) / 1000.0  # Convert ms to seconds
    dt = np.append(dt, dt[-1])  # Same length as data

    # Run AHRS filter

    print("Avg dt:",np.average(dt))

    attitude = ahrs.filters.EKF(acc=acc, gyr=gyr,mag=mag,Dt=np.average(dt),q0= [1,0,0,0])
    quaternions = attitude.Q

    # Remove gravity to get linear acceleration in world frame
    positions = [np.zeros(3)]
    velocities = [np.zeros(3)]

    for i in range(1, len(acc)):
        R = q2R(quaternions[i])      # Rotation matrix from body to world
        lin_acc = R @ acc[i]         # Rotate to world frame
        lin_acc -= [0, 0, 9.81]      # Remove gravity

        v = velocities[-1] + lin_acc * dt[i]
        p = positions[-1] + v * dt[i]

        velocities.append(v)
        positions.append(p)

    positions = np.array(positions)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:,0], positions[:,1], positions[:,2])
    ax.set_title("Estimated Position from IMU")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    plt.show()
