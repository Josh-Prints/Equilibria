# regenerates train.js (the Node multi-core trainer used by GitHub Actions) from sim.js
import os
d=os.path.dirname(os.path.abspath(__file__))
rd=lambda n:open(os.path.join(d,n)).read()
open(os.path.join(d,'train.js'),'w').write(rd('train.head.js')+rd('det_weights.js')+rd('sim.js')+rd('train.tail.js'))
print('wrote train.js')
