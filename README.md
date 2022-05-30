# biphoton_gates_calculator
A tool for generating SymPy expressions and python code of unitary operators that describe linear optical quantum circuits that utilize two photons.

First, the script (json_netlist.py) parses JSON netlist description of the linear optical circuits and converts it into an operator describing single photon mapping. Netlist specifies which element operates on which modes (see examples) and the specific components are declared in a dedicated script file (components.py). In the second step, the single-photon operator is converted into a logical operator that describes the action of the gate in a coincidence basis. The inputs for conversion is the list describing how logical computation states are encoded into the presence of photons in optical modes and similarly a decoding list.

Refer to /examples to see how netlists are created and how logical input/output states are defined.
