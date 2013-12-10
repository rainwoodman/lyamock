def fpga(cosmology, delta, a, Dv=0, uv=0, A0=lambda a: 1.0):
  """uv is photonionization rate in 1e-12/s 
     is a good starting value.
     Dv = d v_r / (H0 d r) is the velocity gradiant along los
      in unit of H0.
     A0(a) is the normalization to align the mean F to
     Faucher-Giguere fit of the observation.
     Generaly one shall run fpga once to obtain 
  """
  beta = 1.65
  Ea = cosmology.Ea(a)

  # black magic from Xiaoying
  # 5.51 is in km/s / Mpc/h
  # H0 is in km/s / Mpc/h
  A1 = numpy.exp(beta * delta)
  A2 = (1 + Dv / Ea) ** -1
  A3 = 1e-12 / uv
  tau = A1 * A2 * A3
  return tau
