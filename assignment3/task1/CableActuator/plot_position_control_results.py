import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Processing the data from the file 
data = pd.read_csv("positions_freq0-002.csv")
data.columns = data.columns.str.strip() # Removing spacing characters 
goal_positions = data[['Goal_Position_X', 'Goal_Position_Y']]
end_effector_positions = data[['Effector_Position_X', 'Effector_Position_Y']]

# Calcuing MSE between goal and effector position 
mae_x = mean_absolute_error(goal_positions['Goal_Position_X'], end_effector_positions['Effector_Position_X'])
mae_y = mean_absolute_error(goal_positions['Goal_Position_Y'], end_effector_positions['Effector_Position_Y'])

# Plotting the positions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(goal_positions['Goal_Position_X'], label='Goal X Positions')
plt.plot(end_effector_positions['Effector_Position_X'], label='Effector X Positions')
plt.title(f'X Position Comparision (MSE: {mae_x:.3f})')
plt.xlabel('Timestep')
plt.ylabel('X Position [mm]')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(goal_positions['Goal_Position_Y'], label='Goal Y Positions')
plt.plot(end_effector_positions['Effector_Position_Y'], label='Effector Y Positions')
plt.title(f'Y Position Comparision (MSE: {mae_y:.3f})')
plt.xlabel('Timestep')
plt.ylabel('Y Position [mm]')
plt.legend()

plt.suptitle('Goal vs Effector Positions (Frequency = 0.002)')
plt.show()
