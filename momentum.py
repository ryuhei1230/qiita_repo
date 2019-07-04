import matplotlib.animation as anm
import matplotlib.pyplot as plt
import numpy as np

abc

X, Y = np.mgrid[-10:11, -10:11]
Z = X**2 + 3 * Y**2
fig, ax = plt.subplots(figsize=(7, 7))
cs = ax.contour(X, Y, Z)
ax.clabel(cs, fontsize=10, fmt='%.2f')
now_x, now_y = 9, 9
diff_x, diff_y = 1, 1   # 最初にupdate_timeで参照する値
learning_rate = 0.01
momentum_x, momentum_y = 0, 0


def update_time():
    t = 0
    global diff_x, diff_y
    while True:
        if (abs(diff_x) < 0.01) and (abs(diff_y) < 0.01):
            break
        yield t
        t += 1


def update(i):
    global now_x,  now_y, momentum_x, momentum_y, diff_x, diff_y
    diff_x = 2 * now_x * learning_rate
    diff_y = 6 * now_y * learning_rate
    new_x = now_x - diff_x + (0.9 * momentum_x)
    new_y = now_y - diff_y + (0.9 * momentum_y)
    plt.title(f'gradient descent\n diff_x: {diff_x:.3f}  diff_y: {diff_y:.3f}'
              f'iteration: {i+1}')
    plt.plot(now_x, now_y, marker='o', markersize=3, color='blue')
    plt.plot([now_x, new_x], [now_y, new_y], color='black')
    momentum_x = new_x - now_x  # 最後に更新
    momentum_y = new_y - now_y
    now_x = new_x
    now_y = new_y


ani = anm.FuncAnimation(fig, update, interval=300, frames=update_time,
                        repeat=False)

