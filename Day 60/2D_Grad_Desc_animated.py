import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to minmize
def f(x):
	return 8 * (x-10) * np.sin(0.5*x-5) + 200

# Derivative of the function
def fd(x):
	return 4 * (x - 10) * np.cos(0.5*x - 5) + 8 * np.sin(0.5*x-5)

x_min = -30
x_max = 30
x = np.linspace(x_min, x_max, 200)
y = f(x)
r = 0.005  # Learning rate
x_est = 25  # Starting point
y_est = f(x_est)

def animate(i):
	global x_est
	global y_est

	# Gradient descent
	x_est = x_est - fd(x_est) * r
	y_est = f(x_est)

	# Update the plot
	scat.set_offsets([[x_est,y_est]])
	text.set_text("Value : %.2f" % y_est)
	line.set_data(x, y)
	return line, scat, text

def init():
	line.set_data([], [])
	return line,

# Visualization Stuff
fig, ax = plt.subplots()
ax.set_xlim([x_min, x_max])
ax.set_ylim([-5, 500])
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
plt.title("Gradient Descent Local Minima")
line, = ax.plot([], [])
scat = ax.scatter([], [], c="red")
text = ax.text(-25,450,"")

ani = animation.FuncAnimation(fig, animate, 40,
	init_func=init, interval=100, blit=True)

plt.show()