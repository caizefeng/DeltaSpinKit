#!/usr/bin/env python3
# @File    : 3DPlotMAE.py
# @Time    : 9/23/2021 2:09 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

theta = np.load("theta.npy")
phi = np.load("phi.npy")
energy = np.load("energy.npy")

phi = np.concatenate([phi, [2 * np.pi]])
energy = np.concatenate([energy, energy[:, 0].reshape(-1, 1)], axis=1)

vector = np.array([-0.519, -0.519, 1.038])

r = np.linalg.norm(vector)
r_hover = 1.02 * r
t, p = np.meshgrid(theta, phi, indexing="ij")
x = r * np.sin(t) * np.cos(p)
y = r * np.sin(t) * np.sin(p)
z = r * np.cos(t)

fig = make_subplots(rows=1, cols=1, specs=[[{"is_3d": True}]], subplot_titles=["MAE"])
fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=energy,
                         colorscale="viridis", showscale=True,
                         colorbar=dict(lenmode='fraction',
                                       len=0.8,
                                       tickvals=np.linspace(energy.min(), energy.max(), 8)
                                       )
                         )
              , 1, 1)

i_max = np.unravel_index(np.argmax(energy), energy.shape)
i_min = np.unravel_index(np.argmin(energy), energy.shape)
t_max = t[i_max]
p_max = p[i_max]
t_min = t[i_min]
p_min = p[i_min]
scatter_t = np.array([t_max, t_min])
scatter_p = np.array([p_max, p_min])

fig.add_trace(go.Scatter3d(x=r_hover * np.sin(scatter_t) * np.cos(scatter_p),
                           y=r_hover * np.sin(scatter_t) * np.sin(scatter_p),
                           z=r_hover * np.cos(scatter_t),
                           name="Maxima and Minima",
                           mode='markers',
                           marker=dict(color="blue",
                                       size=5,
                                       symbol="x"
                                       )
                           ))

vector = np.array([-0.519, -0.519, 1.038])
x, y, z = vector
other_t = math.atan2(math.sqrt(x ** 2 + y ** 2), z)
other_p = math.atan2(y, x)
fig.add_trace(go.Scatter3d(x=[r_hover * np.sin(other_t) * np.cos(other_p)],
                           y=[r_hover * np.sin(other_t) * np.sin(other_p)],
                           z=[r_hover * np.cos(other_t)],
                           name="Another Ni",
                           mode='markers',
                           marker=dict(symbol="x",
                                       color="green",
                                       opacity=1,
                                       size=5
                                       )
                           ))
# fig.add_trace(go.Cone(x=[0], y=[0], z=[0], u=[vector[0]], v=[vector[1]], w=[vector[2]],
#                       anchor="tail",
#                       colorscale='viridis',
#                       showscale=False))

fig.update_layout(title_text="NiO", width=1000, height=1000)
fig.show()
