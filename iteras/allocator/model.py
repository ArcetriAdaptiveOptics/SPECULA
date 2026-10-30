'''
Graph of a simulation, as seen by the allocator: processing objects in
trigger order, the connections between them, and the data objects they
reference.
'''
import json
import inspect
import hashlib
from dataclasses import dataclass

from iteras.base_processing_obj import BaseProcessingObj
from iteras.base_data_obj import BaseDataObj
from iteras.connections import split_output

# Parameters that only say where an object runs: they do not change what it computes
PLACEMENT_KEYS = ('target_rank', 'target_device_idx')
# Parameters that do not change the cost of a step: run length, displays and
# directories (which differ between computers)
MAIN_RUN_KEYS = ('total_time', 'display_server', 'root_dir')
PATH_KEYS = ('store_dir',)


@dataclass(frozen=True)
class Edge:
    src: str        # producer
    out: str        # output name of the producer
    dst: str        # consumer
    delay: int      # 0, or -1 for a value of the previous step


def _hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _ref_names(pars):
    '''Names of the sections referenced by the "_ref", "_list_ref" and "_dict_ref" parameters'''
    names = []
    for name, value in pars.items():
        if not name.endswith('_ref') or value is None:
            continue
        names.extend(value if isinstance(value, (list, tuple)) else [value])
    return names


class SimGraph:
    '''
    Processing objects, connections and references of a simulation.

    Parameters
    ----------
    params : dict
        Simulation parameters, after overrides, without the Architecture section.
    simul : iteras.simul.Simul
        Used for its helpers (trigger order, class lookup).
    '''

    def __init__(self, params, simul):
        self.params = params
        self.classes = {}
        self.main_key = None
        for key in params:
            klass = simul.param_class(params, key)
            if klass is None:
                raise ValueError(f'Allocator: cannot determine the class of section {key}')
            self.classes[key] = klass
            if simul.is_main_params(params, key):
                self.main_key = key

        # Main parameters completed with the defaults of their class (as Simul does)
        simul.setSimulParams(params)
        self.main_params = dict(simul.mainParams or {})

        order, order_idx = simul.build_trigger_order(params)
        self.procs = [name for name in order if issubclass(self.classes[name], BaseProcessingObj)]
        self.level = {name: idx for name, idx in zip(order, order_idx) if name in self.procs}
        self.position = {name: i for i, name in enumerate(self.procs)}
        # Objects whose class does not accept target_device_idx run on the default device
        self.device_settable = {name: 'target_device_idx' in
                                inspect.getfullargspec(self.classes[name].__init__).args
                                for name in self.procs}
        # Objects that split their work over several devices (e.g. DistributedSH)
        self.spread = {name: max(int(self.classes[name].devices_used(params[name])), 1)
                       for name in self.procs}
        self.data_objs = [key for key, klass in self.classes.items()
                          if issubclass(klass, BaseDataObj) and key != self.main_key]

        self.edges = []
        input_data = {name: [] for name in self.procs}     # data objects connected as inputs
        for dst in self.procs:
            for value in params[dst].get('inputs', {}).values():
                for output_name in value if isinstance(value, list) else [value]:
                    out = split_output(output_name)
                    if out.obj_name in self.position:
                        self.edges.append(Edge(out.obj_name, out.output_key, dst, out.delay))
                    elif out.obj_name in self.data_objs:
                        input_data[dst].append(out.obj_name)
                    else:
                        raise ValueError(f'Allocator: input {output_name} of {dst} does not come '
                                         'from a processing object or a data object')
        self.in_edges = {name: [] for name in self.procs}
        self.out_edges = {name: [] for name in self.procs}
        for e in self.edges:
            self.in_edges[e.dst].append(e)
            self.out_edges[e.src].append(e)

        # References: consumer -> data objects, consumer -> processing objects
        self.data_refs = {}
        self.proc_refs = {}
        for name in self.procs:
            refs = [r for r in _ref_names(params[name]) if r != self.main_key]
            self.data_refs[name] = list(dict.fromkeys([r for r in refs if r in self.data_objs]
                                                      + input_data[name]))
            self.proc_refs[name] = [r for r in refs if r in self.position]
        self.ref_consumers = {d: [p for p in self.procs if d in self.data_refs[p]] for d in self.data_objs}
        # Data objects referenced by other data objects must be built where those are
        self.data_data_refs = {d: [r for r in _ref_names(params[d]) if r in self.data_objs]
                               for d in self.data_objs}

        self.pinned = {}
        for name in self.procs:
            pars = params[name]
            if any(k in pars for k in PLACEMENT_KEYS):
                self.pinned[name] = (pars.get('target_rank'), pars.get('target_device_idx'))

        self.keys = self._object_keys()

    def cls(self, name):
        return self.classes[name]

    def section_hash(self, name):
        '''Hash of the parameters of a section that change what it computes'''
        pars = {k: v for k, v in self.params[name].items() if k not in PLACEMENT_KEYS + PATH_KEYS}
        if name == self.main_key:
            pars = {k: v for k, v in pars.items() if k not in MAIN_RUN_KEYS}
        return _hash(pars)

    def _object_keys(self):
        '''
        Key of each processing object for the profile cache: it changes when
        the object, the data objects it references, or the objects upstream
        of it (whose outputs set the size of its inputs) change.
        '''
        keys = {}
        main_hash = self.section_hash(self.main_key) if self.main_key else ''
        for name in self.procs:         # trigger order: producers of delay-0 inputs come first
            upstream = sorted(f'{keys[e.src]}.{e.out}' if e.delay == 0 else f'{e.src}.{e.out}:-1'
                              for e in self.in_edges[name])
            refs = sorted(self.section_hash(r) for r in self.data_refs[name])
            keys[name] = _hash([self.section_hash(name), main_hash, upstream, refs])
        for name in self.data_objs:
            keys[name] = _hash([self.section_hash(name), main_hash])
        return keys

    def signature(self):
        '''Hash of each section, to check whether a saved allocation still applies'''
        return {name: self.section_hash(name) for name in self.params}

    def pins(self):
        return {name: list(pin) for name, pin in sorted(self.pinned.items())}

    def upward_rank(self, weight, comm):
        '''
        Length of the longest path from each object to the end of the step,
        over the delay-0 connections: weight(name) + max over successors of
        (comm(edge) + rank(successor)).
        '''
        rank = {}
        for name in reversed(self.procs):
            succ = [comm(e) + rank[e.dst] for e in self.out_edges[name] if e.delay == 0]
            rank[name] = weight(name) + (max(succ) if succ else 0.0)
        return rank
