from layout import Layout

_layout = Layout(1024,dp_size=8, tp_size=8, pp_size=4, sp_size=2, ep_size=2)

print(_layout.get_pp_ranks())