# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
from visdom import Visdom
import numpy as np

viz = Visdom()


def init_visdom():
    viz.close()
    window = viz.line(X=np.arange(0, 1),
                      Y=np.reshape(0, 1),
                      opts=dict(
                      fillarea=True,
                      legend=False,
                      width=660,
                      height=660,
                      xlabel='Number of weight updates',
                      ylabel='Cost',
                      ytype='lin',
                      title='Loss',
                      marginleft=0,
                      marginright=0,
                      marginbottom=0,
                      margintop=0,)
                      )

    windowB = viz.line(X=np.arange(0, 1),
                  Y=np.reshape(0, 1),
                  opts=dict(
                  fillarea=True,
                  legend=False,
                  width=660,
                  height=660,
                  xlabel='Number of weight updates',
                  ytype='lin',
                  title='Sparsity Term Monitoring',
                  marginleft=0,
                  marginright=0,
                  marginbottom=0,
                  margintop=0,)
                  )

    return window, windowB

# EOF
