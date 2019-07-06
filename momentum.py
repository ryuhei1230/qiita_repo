import matplotlib.animation as anm
import matplotlib.pyplot as plt
import numpy as np
import argparse


def main(args):
    fig = draw_contour()
    location = np.array((args.x_init_location, args.y_init_location))
    animation_gradient = \
        Animation_gradient(location=location, learning_rate=args.learning_rate,
                           min_diff=args.min_diff)
    ani = \
        anm.FuncAnimation(fig, animation_gradient.update_location,
                          interval=300,
                          frames=animation_gradient.update_time,
                          repeat=False)


def draw_contour():
    X, Y = np.mgrid[-10:11, -10:11]
    Z = X**2 + 3 * Y**2
    fig, ax = plt.subplots(figsize=(7, 7))
    cs = ax.contour(X, Y, Z)
    ax.clabel(cs, fontsize=10, fmt='%.2f')
    return fig


class Animation_gradient:
    def __init__(self, location, learning_rate, min_diff):
        self.location = location
        self.diff = np.array((min_diff, min_diff))
        self.learning_rate = learning_rate
        self.momentum = np.array((0, 0))

    def update_time(self):
        t = 0
        while True:
            if all(abs(self.diff) < self.min_diff):
                break
            yield t
            t += 1

    def update_location(self, i):
        self.diff = 2 * self.coordinate * self.learning_rate
        self.coordinate = self.new_cooridinate - self.diff + (
            0.9 * self.momentum)
        plt.title(f'gradient descent\n diff_x:  {self.diff[0]:.3f}'
                  f'diff_y: {self.diff[1]:.3f}  iteration: {i+1}')
        plt.plot(self.x_location, self.y_location,
                 marker='o', markersize=3, color='blue')
        plt.plot([self.x_location, self.new_x_location],
                 [self.y_location, self.new_y_location], color='black')
        self.momentum_x = self.new_location - self.location  # 最後に更新
        self.location = self.new_location


parser = argparse.ArgumentParser()
parser.add_argument('-x', '--x_init_location', default=9, type=int)
parser.add_argument('-y', '--y_init_location', default=9, type=int)
parser.add_argument('-l', '--learning_rate', default=0.01, type=float)
parser.add_argument('-m', '--min_diff', default=0.01, type=float)   # 必須の引数を
args = parser.parse_args()
main(args)
