from random import randint

import mock.mock
from PIL import Image, ImageDraw, ImageFont
import os
# import os.path
import numpy
import numpy as np
from sys import platform
from extensions import *

canvas_size=50 #100
min_size = 8  # 8#12
max_size = 28  # 48
max_padding = canvas_size-max_size
extra_y = 0  # 10 # for oversized letters g,...
sizes = range(min_size, max_size)
if min_size == max_size: sizes = [min_size]
forms = ['circle','line','square']
nForms = len(forms)
max_angle = 30  # 40


styles = ['regular', 'light', 'medium', 'bold']
patterns =[] # textures

from enum import Enum

class Target(Enum):  # labels
	form = 1
	size = 2
	color = 3
	pattern = 4
	position = 5
	style = 6
	angle = 7


nClasses = {
	Target.form: nForms,  # classification
	Target.pattern: len(patterns),  # classification
	Target.style: len(styles),  # classification
	Target.size: 1,  # max_size # regression
	Target.angle: 1,  # max_angle # regression
	Target.position: 2,  # x,y # regression
	Target.color: 3,  # RGB # regression
}

def pos_to_arr(pos):
	# return [pos['x'], pos['y']]
	return pos[0]#, pos['y']]


class batch():
	def __init__(self, batch_size=64, target=Target.size):
		self.batch_size = batch_size
		self.target = target
		self.shape = [max_size * max_size + extra_y, nClasses[target]]
		# self.shape=[batch_size,max_size,max_size,len(letters)]
		self.train = self
		self.test = self
		self.test.images, self.test.labels = self.next_batch()

	def next(self, batch_size=None):
		return self.next_batch()

	def next_batch(self, batch_size=None):
		shapes = [shape() for i in range(batch_size or self.batch_size)]
		xs = map(lambda x: x.matrix() / 255., shapes)
		if self.target == Target.form: ys = [one_hot(s.ord, nForms, 0) for s in shapes]
		elif self.target == Target.size: ys = [s.size for s in shapes]
		elif self.target == Target.position: ys = [pos_to_arr(s.pos) for s in shapes]
		else: raise Exception("Target not yet implemented: %s",self.target)
		return list(xs), list(ys)


def pick(xs,default=None):
	l = len(xs)
	if l==0: return default
	return xs[randint(0, l - 1)]


def one_hot(item, num_classes, offset):
	labels_one_hot = numpy.zeros(num_classes)
	labels_one_hot[item - offset] = 1
	return labels_one_hot


def getColor():
	return 'black'  # 'white'#self.random_color()


def getPosition(size):
	fro = pick(range(0, max_padding)), pick(range(0, max_padding))
	# to = pick(range(fro[0], max_padding- fro[0])), pick(range(fro[1], max_padding- fro[1]))
	# to = fro[0]+ size,fro[1]+size
	return fro#, to]


# return {'x': pick(range(0, max_padding)), 'y': pick(range(0, max_padding))}


class shape():

	def __init__(self, *margs, **args):  # optional arguments
		self.pattern = None
		if not args:
			if margs:
				args = margs[0]  # ruby style hash args
			else:
				args = {}
		self.form = args['form'] if 'form' in args else pick(forms)
		self.size = args['size'] if 'size' in args else pick(sizes)
		self.pattern = args['pattern'] if 'pattern' in args else pick(patterns)
		self.pos = args['pos'] if 'pos' in args else getPosition(self.size)
		self.angle = args['angle'] if 'angle' in args else 0  # pick(range(-max_angle,max_angle))
		self.color = args['color'] if 'color' in args else getColor()
		self.style = args['style'] if 'style' in args else self.get_style()
		self.background = args['back'] if 'back' in args else None  # self.random_color() # 'white' #None #pick(range(-90, 180))

	# self.padding = self.pos

	def projection(self):
		return self.matrix(), self.ord

	def random_color(self):
		r = randint(0, 255)
		g = randint(0, 255)
		b = randint(0, 255)
		a = randint(0, 255)
		return (r, g, b, a)

	def get_style(self):
		return 'regular'

	def matrix(self):
		try:
			return np.array(self.image())
		except:
			return np.array(canvas_size * canvas_size)


	def image(self):
		fill='black' #None
		outline=None
		fro = self.pos
		to = fro[0]+ self.size,fro[1]+ self.size
		xy = [fro,to]
		size = (canvas_size, canvas_size)
		if self.background:
			img = Image.new('RGBA', size, self.background)  # background_color
		else:
			img = Image.new('L', size, 'white')  # # grey
		draw = ImageDraw.Draw(img)
		# draw.text((padding['x'], padding['y']), text, font=ttf_font, fill=self.color)
		draw.rectangle(xy,fill,outline)
		if (self.angle > 0 or self.angle < 0) and self.size > 20:
			rot = img.rotate(self.angle, expand=1).resize(size)
			if self.background:
				img = Image.new('RGBA', size, self.background)  # background_color
			else:
				img = Image.new('L', size, '#FFF')  # FUCK BUG! 'white')#,'grey')  # # grey
			img.paste(rot, (0, 0), rot)
		return img

	def show(self):
		self.image().show()

	@classmethod
	def random(cls):
		l = shape()
		l.size = pick(sizes)
		l.form = pick(forms)
		l.pos = (pick(range(0, 10)), pick(range(0, 10)))
		l.style = pick(styles)  # None #
		l.pattern= pick(patterns)
		l.angle = 0

	def __str__(self):
		format = "shape{form='%s',size=%d,pos='%s',angle=%d,color=%s}"
		return format % (self.form, self.size,  self.pos, self.angle, self.color)#, self.pattern)

	# def print(self):
	# 	print(self.__str__)


# @classmethod	# can access class cls
# def ls(cls, mypath=None):

# @staticmethod	# CAN'T access class
# def ls(mypath):

if __name__ == "__main__":
	l = shape()
	m = l.matrix()
	print(l)
	l.show()


