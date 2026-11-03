'''
Text report of an allocation.
'''

from .cost import split_devices
from .machine import GiB

MiB = 1024 ** 2


def make_report(graph, machine, est, solution, profile, allocator, allocation, by_nodes=()):
    placement = solution.placement()
    ev = solution.evaluation
    lines = ['Resources:', machine.describe(), '']
    lines.append(f'Layout: {solution.layout.policy}, {allocation["n_nodes"]} node(s), '
                 f'{allocation["n_ranks"]} MPI rank(s)')
    for p in allocation['processes']:
        lines.append(f'  rank {p["rank"]}: node {p["node"]}, devices {p["devices"]}')
    lines.append('')
    if by_nodes:
        lines.append('Estimated time per step with the first k nodes: ' +
                     ', '.join(f'{k}: {t / 1e6:.2f} ms' for k, t in by_nodes))
    lines.append(f'Estimated time per step: {ev.step_ns / 1e6:.2f} ms '
                 f'(data copied between devices: {ev.comm_bytes / MiB:.1f} MiB per step)')
    calib = getattr(est, 'calibration', None)
    if calib:
        lines.append(f'Profiling run (all objects on one GPU): measured {calib[1] / 1e6:.2f} ms per step, '
                     f'model {calib[0] / 1e6:.2f} ms; loop overhead added per step and process: '
                     f'{est.overhead_ns / 1e6:.2f} ms')
    lines.append('')
    if solution.split:
        lines.append('Objects split over several GPUs: ' + ', '.join(
            f'{name} ({allocation["objects"][name].get("class", graph.cls(name).__name__)}, {k} GPUs)'
            for name, k in solution.split.items()))
    lines.append(memory_table(machine, ev.memory))
    lines.append('')
    lines.append(object_table(graph, machine, est, placement, allocation, solution.split))
    estimated = sorted({n for (n, k), c in est.costs.items()
                        if not c.measured and any(placement[n][1] == s for s, sl in enumerate(machine.slots)
                                                  if sl.kind == k)})
    if estimated:
        lines.append('')
        lines.append('GPU times scaled from another GPU model (relative_speed) for: '
                     + ', '.join(estimated))
    return '\n'.join(lines) + '\n'


def memory_table(machine, memory):
    lines = ['Memory per device (GiB, used / available):']
    for s, slot in enumerate(machine.slots):
        if s in memory:
            lines.append(f'  {slot.label():16s} {memory[s] / GiB:8.2f} / {slot.memory / GiB:8.2f}')
    return '\n'.join(lines)


def object_table(graph, machine, est, placement, allocation=None, split=None):
    '''
    One line per object. For objects split over several GPUs (in the parameter
    file, or by the allocator: *split*) the class is that of the split object
    and the devices are all those it uses; the costs are those of the whole object.
    '''
    spread = {**graph.spread, **(split or {})}
    lines = [f'{"object":24s} {"class":22s} {"rank":>4s} {"device":>7s} '
             f'{"host ms":>8s} {"GPU ms":>8s} {"mem MiB":>9s}']
    for name in graph.procs:
        if name not in placement:
            continue
        p, s = placement[name]
        c = est.cost(name, s)
        devices = ','.join(str(machine.slots[d].idx)
                           for d in split_devices(machine, s, spread.get(name, 1)))
        obj = allocation['objects'].get(name, {}) if allocation else {}
        klass = obj.get('class', graph.cls(name).__name__)
        rank = obj.get('target_rank', 0) if allocation else p.rank
        lines.append(f'{name[:24]:24s} {klass[:22]:22s} {rank:4d} {devices:>7s} '
                     f'{c.host / 1e6:8.3f} {c.g / 1e6:8.3f} {c.mem / MiB:9.1f}')
    return '\n'.join(lines)

