This is a re-implementation (in pytorch) of part of the AAAI16 paper: 'Visual Learning of Arithmetic Operations'. It is based on (forked from) the original repo: [[here]](https://github.com/Yedid/arithmetic).

Requirements: PIL, matplotlib, numpy, pytorch

Running:

Train: `python zrun.sh --mode train --oper [ADD|MUL]`

Test: `python zrun.sh --mode test --oper [ADD|MUL]`

Salience-Visualize: `python zrun.sh --mode salience --oper [ADD|MUL]`
