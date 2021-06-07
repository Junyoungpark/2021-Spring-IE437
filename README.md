# 2021-Spring-IE437

This repository contains several types of gradient-based optimization problems that utilize the automatic differentiation tools (e.g. `pytorch`)  for modeling objective, constraints in the optimization problems.

## Dependencies

```
numpy
pytorch (>=1.5)
scipy
```

Note) We utilize `torch.autograd.functional` API to compute the hessian and jacobian of `torch` functions. This feature
is experimentally supported by `torch >= 1.5.0`.

## Tutorials

### Tutorial 1. `torch` model in cost function

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{120}&space;\begin{aligned}&space;\min_{x}&space;&\,&space;f_\theta(x)&space;\\&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\dpi{120}&space;\begin{aligned}&space;\min_{x}&space;&\,&space;f_\theta(x)&space;\\&space;\end{aligned}" title="\begin{aligned} \min_{x} &\, f_\theta(x) \\ \end{aligned}" /></a>

When we deal with the optimization problems that do not facilitate constraints, we 'can' solve the problems (not
necessarily mean that we can achieve the global optimum) with vanilla gradient descent methods such as gradient
descent (GD) or Adam.

### Tutorial 2. `torch` model in cost function + Box constraints

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{array}{l}&space;\min&space;_{x}&space;f_\theta(x)&space;\\&space;\text&space;{&space;s.t.&space;}&space;x_{\min&space;}&space;\leq&space;x&space;\leq&space;x_{\max&space;}&space;\end{array}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{array}{l}&space;\min&space;_{x}&space;f_\theta(x)&space;\\&space;\text&space;{&space;s.t.&space;}&space;x_{\min&space;}&space;\leq&space;x&space;\leq&space;x_{\max&space;}&space;\end{array}" title="\begin{array}{l} \min _{x} f_\theta(x) \\ \text { s.t. } x_{\min } \leq x \leq x_{\max } \end{array}" /></a>

### Tutorial 3. MPC with `torch` model + Box constraints

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\min&space;_{\left\{u_{t}\right\}_{t=0,&space;\ldots,&space;H}}&space;\sum&space;&x_{t}^{T}&space;Q&space;x_{t}&plus;u_{t}^{T}&space;R&space;u_{t}\\&space;\text&space;{s.t.}&space;\quad&space;x_{t&plus;1}&=f_{\theta}\left(x_{t},&space;u_{t}\right)\\&space;u_{\min&space;}&space;&\leq&space;u_t&space;\leq&space;u_{\max&space;}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\min&space;_{\left\{u_{t}\right\}_{t=0,&space;\ldots,&space;H}}&space;\sum&space;&x_{t}^{T}&space;Q&space;x_{t}&plus;u_{t}^{T}&space;R&space;u_{t}\\&space;\text&space;{s.t.}&space;\quad&space;x_{t&plus;1}&=f_{\theta}\left(x_{t},&space;u_{t}\right)\\&space;u_{\min&space;}&space;&\leq&space;u_t&space;\leq&space;u_{\max&space;}&space;\end{aligned}" title="\begin{aligned} \min _{\left\{u_{t}\right\}_{t=0, \ldots, H}} \sum &x_{t}^{T} Q x_{t}+u_{t}^{T} R u_{t}\\ \text {s.t.} \quad x_{t+1}&=f_{\theta}\left(x_{t}, u_{t}\right)\\ u_{\min } &\leq u_t \leq u_{\max } \end{aligned}" /></a>

