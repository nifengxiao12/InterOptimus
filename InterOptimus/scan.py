from optimize import WorkPatcher
from fireworks import LaunchPad

my_wp = WorkPatcher.from_dir('.')
my_wp.param_parse('itop_test',1e-4, 5,vacuum_over_film=15)
my_wp.get_unique_terminations(0)
wfs = my_wp.PatchRegistrationStaticScanWorkFlow(0, 0, 20, NCORE = 8)
lp = LaunchPad.auto_load()
lp.add_wf(wfs)
