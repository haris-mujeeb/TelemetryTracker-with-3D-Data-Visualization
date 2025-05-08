import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#check https://github.com/balamuruganky/EKF_IMU_GPS/tree/master

data = pd.read_csv('test_data.csv')
print(data.head()) # to display the first 5 lines of loaded data
data["timestamp"] = data["timestamp"]- data["timestamp"][0]
data["IMU-ticks"] = data["IMU-ticks"]- data["IMU-ticks"][0]
data["GPS-ticks"] = data["GPS-ticks"]- data["GPS-ticks"][0]
data["pressure-ticks"] = data["pressure-ticks"]- data["pressure-ticks"][0]

data['lat'] = data['lat'] / 1e7
data['long'] = data['long'] / 1e7

print(data)



colors = data['GPS-ticks']
fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(10, 6))
sc = ax[0,0].scatter(data['lat'], data['long'], c=colors, cmap='viridis', s=10)
ax[0,1].scatter(data["IMU-ticks"], data['aX'],c=data["IMU-ticks"],cmap='viridis', s=10)
ax[0,2].scatter(data["IMU-ticks"], data['aY'],c=data["IMU-ticks"],cmap='viridis', s=10)
ax[1,0].scatter(data["IMU-ticks"], data['aZ'],c=data["IMU-ticks"],cmap='viridis', s=10)
ax[1,1].scatter(data["IMU-ticks"], data['gX'],c=data["IMU-ticks"],cmap='viridis', s=10)
ax[1,2].scatter(data["IMU-ticks"], data['gY'],c=data["IMU-ticks"],cmap='viridis', s=10)
ax[2,0].scatter(data["IMU-ticks"], data['gZ'],c=data["IMU-ticks"],cmap='viridis', s=10)


ax[2,1].scatter(data["IMU-ticks"], (data['aX']**2 + data['aX']**2 + data['aZ']**2)**(1/2),c=data["IMU-ticks"],cmap='viridis', s=10)

# Add colorbar for reference
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('GPS Ticks')

plt.show()

